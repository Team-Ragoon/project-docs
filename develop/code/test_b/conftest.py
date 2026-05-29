"""
pytest 공통 설정 파일 (test_b/ 전용)
- 이 파일은 test_b/ 하위에 위치하므로
  sys.path에 상위 디렉터리(develop/code/)를 추가해야
  answer_parser, rag_qna_multi 등을 import할 수 있음
"""
import sys
import os
from collections import defaultdict

# test_b/ → 상위 → develop/code/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 테스트 함수별 결과 수집용 딕셔너리
_results_by_func: dict = defaultdict(lambda: {"passed": 0, "failed": 0})


def _get_func_name(nodeid: str) -> str:
    """
    nodeid에서 테스트 함수 이름만 추출.
    예) test_scenario_b.py::test_flow_b_scenario[타이레놀...] → test_flow_b_scenario
    """
    parts = nodeid.split("::")
    func_part = parts[1] if len(parts) > 1 else parts[0]
    return func_part.split("[")[0]  # 파라미터 부분 제거


def pytest_runtest_logreport(report):
    """각 테스트 결과를 함수별로 수집 (call 단계만 집계)"""
    func_name = _get_func_name(report.nodeid)

    if report.when == "call":
        if report.passed:
            _results_by_func[func_name]["passed"] += 1
        elif report.failed:
            _results_by_func[func_name]["failed"] += 1
    elif report.when == "setup" and report.failed:
        _results_by_func[func_name]["failed"] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """테스트 종료 후 함수별 passed/failed 수와 비율을 출력"""
    if not _results_by_func:
        return

    print("\n" + "=" * 55)
    print("  📊 테스트 결과 요약")
    print("=" * 55)

    total_all = passed_all = failed_all = 0

    for func_name in sorted(_results_by_func):
        counts = _results_by_func[func_name]
        passed = counts["passed"]
        failed = counts["failed"]
        total  = passed + failed
        if total == 0:
            continue

        pass_pct = passed / total * 100
        fail_pct = failed / total * 100

        print(f"\n  🔹 {func_name}")
        print(f"     전체      : {total}개")
        print(f"     ✅ PASSED : {passed}개  ({pass_pct:.1f}%)")
        print(f"     ❌ FAILED : {failed}개  ({fail_pct:.1f}%)")

        total_all  += total
        passed_all += passed
        failed_all += failed

    # 함수가 2개 이상일 때만 합계 출력
    if len(_results_by_func) > 1:
        pass_pct = passed_all / total_all * 100 if total_all else 0
        fail_pct = failed_all / total_all * 100 if total_all else 0
        print(f"\n  {'─' * 45}")
        print(f"  합계")
        print(f"     전체      : {total_all}개")
        print(f"     ✅ PASSED : {passed_all}개  ({pass_pct:.1f}%)")
        print(f"     ❌ FAILED : {failed_all}개  ({fail_pct:.1f}%)")

    print("=" * 55)