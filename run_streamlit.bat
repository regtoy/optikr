@echo off
setlocal

REM Proje köküne geç
cd /d %~dp0

REM Sanal ortam yoksa oluştur
if not exist .venv (
    echo [INFO] .venv olusturuluyor...
    py -m venv .venv
)

REM Sanal ortami aktif et
call .venv\Scripts\activate

REM Pip guncelle ve bagimliliklari kur
echo [INFO] Bagimliliklar yukleniyor...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Streamlit arayuzunu baslat
echo [INFO] Streamlit baslatiliyor...
streamlit run streamlit_app.py

endlocal
