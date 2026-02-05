# DearLook

사용자 입력 사진에서 **인물만 픽셀화**하여 PNG로 반환하는 AI 파이프라인

## 구조

- `ai/` PyTorch 기반 인물 분리 + 픽셀화 파이프라인
- `backend/` Spring Boot API (참고용)
- `frontend/` Flutter 앱 (참고용)
- `db/` MySQL 스키마/마이그레이션 (참고용)
- `docs/` 문서

## 로컬 개발 (AI)

- Python + PyTorch (CPU-only)
- 인물 세그멘테이션 모델 연동 후 추론
