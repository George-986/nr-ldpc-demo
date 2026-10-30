"""JSON 模板驱动流程的解析、运行与 CLI 测试。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from nr_pusch_ldpc_rx.app.pipeline import (
    PUSCHCase,
    load_config,
    parse_case,
    parse_config,
    run_case,
    run_config,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"


def _minimal(**overrides):
    """最小合法用例字典，便于逐个覆盖字段。"""
    case = {
        "name": "min",
        "transport_block_size": 100,
        "target_rate": 0.3,
        "scrambling": {"n_rnti": 1, "n_id": 0},
    }
    case.update(overrides)
    return case


class TestParseConfig(unittest.TestCase):
    def test_defaults(self):
        """只给必填字段时，其余参数应有明确默认值。"""
        case = parse_case(
            {"transport_block_size": 100, "target_rate": 0.3}
        )
        self.assertIsInstance(case, PUSCHCase)
        self.assertEqual(case.name, "case")
        self.assertEqual(case.modulation, "QPSK")
        self.assertEqual(case.qm, 2)
        self.assertEqual(case.rv, 0)
        self.assertEqual(case.n_layers, 1)
        self.assertIsNone(case.num_coded_bits)
        self.assertIsNone(case.base_graph)
        self.assertIsNone(case.scrambling)
        self.assertEqual(case.tb_source, "random")
        self.assertEqual(case.tb_seed, 0)
        self.assertEqual(case.max_ldpc_iter, 30)
        self.assertAlmostEqual(case.alpha, 0.8)
        self.assertTrue(case.early_stop)
        self.assertEqual(case.channel, "none")

    def test_description_metadata(self):
        """description 是允许的元数据，会进入报告；类型错误仍然报错。"""
        case = parse_case(_minimal(description="hello"))
        self.assertEqual(case.description, "hello")
        report = run_config({"description": "top", "cases": [dict(
            _minimal(description="inner"), )]})
        self.assertEqual(report.cases[0].description, "inner")
        with self.assertRaises(TypeError):
            parse_case(_minimal(description=123))

    def test_single_object_and_cases_list(self):
        """支持单用例对象和 {"cases": [...]} 两种顶层形式。"""
        single = parse_config({"transport_block_size": 100, "target_rate": 0.3})
        self.assertEqual(len(single), 1)

        multi = parse_config(
            {
                "cases": [
                    _minimal(name="a"),
                    _minimal(name="b", transport_block_size=200),
                ]
            }
        )
        self.assertEqual([c.name for c in multi], ["a", "b"])
        self.assertEqual(multi[1].transport_block_size, 200)

    def test_unknown_keys_rejected(self):
        """拼写错误必须报错，而不是静默忽略。"""
        with self.assertRaisesRegex(ValueError, "target_rat"):
            parse_case(_minimal(target_rat=0.5))
        with self.assertRaisesRegex(ValueError, "cases\\[1\\]"):
            parse_config({"cases": [_minimal(), _minimal(foo=1)]})
        with self.assertRaisesRegex(ValueError, "config"):
            parse_config({"cases": [_minimal()], "extra": 1})
        with self.assertRaisesRegex(ValueError, "scrambling"):
            parse_case(_minimal(scrambling={"n_rnti": 1, "nid": 2}))
        with self.assertRaisesRegex(ValueError, "ldpc"):
            parse_case(_minimal(ldpc={"iter": 10}))
        with self.assertRaisesRegex(ValueError, "channel"):
            parse_case(_minimal(channel={"type": "rayleigh"}))

    def test_missing_required(self):
        with self.assertRaisesRegex(ValueError, "transport_block_size"):
            parse_case({"target_rate": 0.3})
        with self.assertRaisesRegex(ValueError, "target_rate"):
            parse_case({"transport_block_size": 100})

    def test_range_and_type_validation(self):
        with self.assertRaises(ValueError):
            parse_case(_minimal(target_rate=0.0))
        with self.assertRaises(ValueError):
            parse_case(_minimal(target_rate=1.5))
        with self.assertRaises(ValueError):
            parse_case(_minimal(rv=4))
        with self.assertRaises(ValueError):
            parse_case(_minimal(n_layers=0))
        with self.assertRaises(ValueError):
            parse_case(_minimal(modulation="1024QAM"))
        with self.assertRaises(ValueError):
            parse_case(_minimal(base_graph=3))
        with self.assertRaises(TypeError):
            parse_case(_minimal(transport_block_size="100"))
        with self.assertRaises(ValueError):
            parse_case(_minimal(name=""))

    def test_num_coded_bits_multiple(self):
        """G 必须是 Qm*N_L 的整数倍；否则 E_r 无法均分。"""
        parse_case(_minimal(num_coded_bits=388))
        with self.assertRaises(ValueError):
            parse_case(_minimal(num_coded_bits=387))
        with self.assertRaises(ValueError):
            parse_case(_minimal(num_coded_bits=0))
        parse_case(_minimal(num_coded_bits=100, n_layers=2))

    def test_transport_block_bits(self):
        """TB 来源解析：显式比特必须与 A 等长。"""
        case = parse_case(
            _minimal(
                transport_block_size=8,
                transport_block_bits={
                    "source": "explicit",
                    "bits": [1, 0, 1, 1, 0, 0, 1, 0],
                },
            )
        )
        self.assertEqual(case.tb_source, "explicit")
        self.assertEqual(len(case.tb_bits), 8)

        with self.assertRaisesRegex(ValueError, "长度"):
            parse_case(
                _minimal(
                    transport_block_size=8,
                    transport_block_bits={"source": "explicit", "bits": [1, 0]},
                )
            )
        with self.assertRaisesRegex(ValueError, "explicit"):
            parse_case(
                _minimal(transport_block_bits={"source": "explicit"})
            )
        with self.assertRaisesRegex(ValueError, "bits"):
            parse_case(
                _minimal(
                    transport_block_bits={
                        "source": "random",
                        "bits": [0, 1, 0, 1],
                    }
                )
            )
        with self.assertRaises(ValueError):
            parse_case(
                _minimal(
                    transport_block_size=8,
                    transport_block_bits={
                        "source": "explicit",
                        "bits": [0, 2, 0, 1, 0, 1, 0, 1],
                    },
                )
            )

    def test_scrambling_and_msg_a(self):
        case = parse_case(_minimal())
        self.assertEqual(case.scrambling.n_rnti, 1)
        self.assertEqual(case.scrambling.c_init, 2**15)

        msg_a = parse_case(
            _minimal(
                scrambling={
                    "n_rnti": 673,
                    "n_id": 9,
                    "msg_a": True,
                    "n_rapid": 17,
                }
            )
        )
        self.assertEqual(msg_a.scrambling.c_init, 673 * 2**16 + 17 * 2**10 + 9)

        with self.assertRaises(ValueError):
            parse_case(_minimal(scrambling={"n_rnti": 1, "msg_a": True}))

    def test_channel_validation(self):
        case = parse_case(
            _minimal(channel={"type": "awgn", "ebn0_db": 3.0, "seed": 2})
        )
        self.assertEqual(case.channel, "awgn")
        self.assertEqual(case.ebn0_db, 3.0)
        self.assertEqual(case.channel_seed, 2)
        with self.assertRaises(ValueError):
            parse_case(_minimal(channel={"type": "none", "ebn0_db": 3.0}))

    def test_empty_cases(self):
        with self.assertRaises(ValueError):
            parse_config({"cases": []})

    def test_load_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"cases": [_minimal(), _minimal(name="two")]}),
                encoding="utf-8",
            )
            cases = load_config(path)
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[1].name, "two")

        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/path.json")

    def test_load_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


class TestRunConfig(unittest.TestCase):
    def test_two_cases_pass(self):
        report = run_config(
            {
                "cases": [
                    _minimal(name="one", transport_block_size=100),
                    _minimal(name="two", transport_block_size=545),
                ]
            }
        )
        self.assertEqual(len(report.cases), 2)
        self.assertTrue(report.all_passed)
        self.assertEqual(report.passed, 2)
        self.assertEqual(report.failed, 0)
        for case in report.cases:
            self.assertTrue(case.tb_crc_ok)
            self.assertTrue(case.cb_crc_ok)
            self.assertTrue(case.ldpc_converged)
            self.assertEqual(case.bit_errors, 0)
            self.assertGreater(case.decode_seconds, 0.0)

    def test_report_dict(self):
        report = run_config(_minimal(name="dict_case"))
        data = report.as_dict()
        self.assertEqual(data["source"], "<dict>")
        self.assertEqual(data["summary"], {"total": 1, "passed": 1, "failed": 0})
        case = data["cases"][0]
        self.assertEqual(case["name"], "dict_case")
        self.assertTrue(case["passed"])
        self.assertEqual(case["plan"]["base_graph"], 2)
        self.assertEqual(case["plan"]["c"], 1)
        self.assertEqual(case["scrambling"]["c_init"], 2**15)
        self.assertTrue(case["decode"]["tb_crc_ok"])
        # 报告必须可直接 json 序列化。
        json.dumps(data, ensure_ascii=False)

    def test_list_input(self):
        report = run_config([_minimal(name="list_case")])
        self.assertEqual(report.source, "<list>")
        self.assertTrue(report.all_passed)

    def test_tb_sources(self):
        """zeros/ones/alternating 三种确定性来源都应能往返。"""
        for source in ("zeros", "ones", "alternating"):
            with self.subTest(source=source):
                report = run_config(
                    _minimal(
                        transport_block_bits={"source": source},
                        num_coded_bits=388,
                    )
                )
                self.assertTrue(report.all_passed)

    def test_explicit_bits_roundtrip(self):
        bits = [1, 0, 1, 1, 0, 0, 1, 0]
        case = parse_case(
            _minimal(
                transport_block_size=8,
                transport_block_bits={"source": "explicit", "bits": bits},
            )
        )
        result = run_case(case)
        self.assertTrue(result.passed)
        self.assertEqual(result.bit_errors, 0)

    def test_random_seed_is_deterministic(self):
        report_a = run_config(_minimal(name="seed", num_coded_bits=388))
        report_b = run_config(_minimal(name="seed", num_coded_bits=388))
        self.assertEqual(report_a.cases[0].iterations, report_b.cases[0].iterations)
        self.assertTrue(report_a.all_passed and report_b.all_passed)

    def test_awgn_failure_reported(self):
        """极低信噪比下 CRC 失败，报告应给出 failed 而不是抛异常。"""
        report = run_config(
            _minimal(
                name="low_snr",
                transport_block_size=1000,
                num_coded_bits=3390,
                channel={"type": "awgn", "ebn0_db": -6.0, "seed": 1},
            )
        )
        self.assertFalse(report.all_passed)
        self.assertEqual(report.failed, 1)
        self.assertFalse(report.cases[0].tb_crc_ok)
        self.assertGreater(report.cases[0].bit_errors, 0)

    def test_forced_base_graph(self):
        report = run_config(
            _minimal(transport_block_size=1000, base_graph=1)
        )
        self.assertEqual(report.cases[0].base_graph, 1)
        self.assertTrue(report.all_passed)


class TestTemplates(unittest.TestCase):
    def test_example_template(self):
        path = TEMPLATES / "pusch_example.json"
        self.assertTrue(path.is_file())
        report = run_config(path)
        self.assertGreaterEqual(len(report.cases), 5)
        self.assertTrue(report.all_passed)
        # 覆盖 BG1/BG2、多码块、rv、msgA 和 AWGN。
        self.assertEqual({c.base_graph for c in report.cases}, {1, 2})
        self.assertTrue(any(c.code_blocks > 1 for c in report.cases))
        self.assertTrue(any(c.c_init for c in report.cases))
        self.assertEqual(len({c.name for c in report.cases}), len(report.cases))

    def test_single_template(self):
        path = TEMPLATES / "pusch_template.json"
        self.assertTrue(path.is_file())
        cases = load_config(path)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].tb_source, "alternating")
        self.assertTrue(run_config(path).all_passed)


class TestCLI(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "run_pusch.py"), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )

    def test_example_template_cli(self):
        proc = self._run("templates/pusch_example.json", "--quiet")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("6/6", proc.stdout)

    def test_json_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            proc = self._run(
                "templates/pusch_template.json",
                "--quiet",
                "--json-out",
                str(out),
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["passed"], 1)
            self.assertTrue(data["cases"][0]["passed"])

    def test_invalid_config_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps({"transport_block_size": 100, "target_rate": 0.3,
                            "unknown": 1}),
                encoding="utf-8",
            )
            proc = self._run(str(path))
            self.assertEqual(proc.returncode, 2)
            self.assertIn("配置错误", proc.stderr)

    def test_failing_case_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fail.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "fail",
                        "transport_block_size": 1000,
                        "target_rate": 0.3,
                        "num_coded_bits": 3390,
                        "channel": {
                            "type": "awgn",
                            "ebn0_db": -6.0,
                            "seed": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            proc = self._run(str(path), "--quiet")
            self.assertEqual(proc.returncode, 1)
            self.assertIn("0/1", proc.stdout)


if __name__ == "__main__":
    unittest.main()
