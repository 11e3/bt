# PowerShell 프로필 함수 설정 가이드

## check-all 함수를 PowerShell 프로필에 추가하기

### 1. PowerShell 프로필 열기

```powershell
# PowerShell 프로필 경로 확인
$PROFILE

# 프로필 파일이 없으면 생성
if (!(Test-Path -Path $PROFILE)) {
    New-Item -ItemType File -Path $PROFILE -Force
}

# 프로필 파일 열기
notepad $PROFILE
```

### 2. 프로필에 check-all 함수 추가

프로필 파일에 다음 내용을 추가하세요:

```powershell
function check-all {
    <#
    .SYNOPSIS
    코드 품질 검사를 수행합니다.
    
    .DESCRIPTION
    ruff 포맷팅, 린팅, mypy 타입 체크, pytest 테스트를 순차적으로 실행합니다.
    #>
    
    Write-Host "🔍 Running code quality checks..." -ForegroundColor Cyan
    
    Write-Host "`n📝 Formatting code with ruff..." -ForegroundColor Yellow
    ruff format .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Formatting failed" -ForegroundColor Red
        return
    }
    
    Write-Host "`n🔧 Linting and fixing with ruff..." -ForegroundColor Yellow
    ruff check . --fix --unsafe-fixes
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Linting failed" -ForegroundColor Red
        return
    }
    
    Write-Host "`n🔍 Type checking with mypy..." -ForegroundColor Yellow
    mypy src/bt --strict
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Type checking failed" -ForegroundColor Red
        return
    }
    
    Write-Host "`n🧪 Running tests with coverage..." -ForegroundColor Yellow
    pytest --cov=src/bt --cov-report=term-missing
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Tests failed" -ForegroundColor Red
        return
    }
    
    Write-Host "`n✅ All checks passed!" -ForegroundColor Green
}
```

### 3. 프로필 저장 및 재로드

```powershell
# 프로필 재로드
. $PROFILE

# 또는 새 PowerShell 세션 시작
```

### 4. 사용 방법

```powershell
# bt 프로젝트 디렉토리로 이동
cd C:\workspace\dev\bt

# 코드 품질 검사 실행
check-all
```

## 대안: 프로젝트별 스크립트 사용

프로필 함수 대신 프로젝트에 포함된 `check.ps1` 스크립트를 사용할 수도 있습니다:

```powershell
# 실행 권한 설정 (최초 1회만)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 스크립트 실행
.\check.ps1
```

## 개별 도구 실행

필요시 개별 도구를 따로 실행할 수 있습니다:

```powershell
# 포맷팅만
ruff format .

# 린팅만
ruff check . --fix

# 타입 체크만
mypy src/bt --strict

# 테스트만
pytest

# 커버리지 포함 테스트
pytest --cov=src/bt --cov-report=html
```

## VS Code 통합

VS Code에서 작업하는 경우 `.vscode/tasks.json`에 태스크를 추가할 수도 있습니다:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "check-all",
            "type": "shell",
            "command": "${workspaceFolder}/check.ps1",
            "problemMatcher": [],
            "group": {
                "kind": "test",
                "isDefault": true
            }
        }
    ]
}
```

그러면 `Ctrl+Shift+B`로 실행할 수 있습니다.
