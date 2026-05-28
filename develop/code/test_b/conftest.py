"""
pytest 공통 설정 파일 (test_b/ 전용)
- 이 파일은 test_b/ 하위에 위치하므로
  sys.path에 상위 디렉터리(develop/code/)를 추가해야
  answer_parser, rag_qna_multi 등을 import할 수 있음
"""
import sys
import os

# test_b/ → 상위 → develop/code/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))