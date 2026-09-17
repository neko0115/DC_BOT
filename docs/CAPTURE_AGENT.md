# Moxue Capture Agent

`Moxue Capture Agent` 將遊戲畫面與裝置校準從 Discord Bot 本體拆開。哪一台 Windows 電腦實際執行遊戲，哪一台就保存自己的聊天室 ROI Profile。

## 架構

桌機／Server：

```text
Discord Bot / Tool Gateway   127.0.0.1:8765
Moxue Capture Hub            ws://<server>:8878/capture/ws
```

遊戲電腦：

```text
MoxueCapture
├─ Local Capture API         127.0.0.1:8877
├─ DXcam / ImageGrab
├─ Calibration GUI
├─ OCR / Profiles
└─ outbound WebSocket ─────→ Moxue Capture Hub
```

遠端 Agent 是**主動連回 Server**，Server 不需要主動連進玩家電腦，也不需要玩家在 Router 上做 port forwarding。

目前 WebSocket transport 是 `ws://`，適合可信任 LAN / VPN 測試。不要直接把 8878 暴露到公開 Internet；正式跨網路部署需要再加入 TLS (`wss://`) 或透過可信任 VPN。

## Server 設定

`.env`：

```env
CAPTURE_HUB_ENABLED=true
CAPTURE_HUB_HOST=0.0.0.0
CAPTURE_HUB_PORT=8878
```

`0.0.0.0` 代表讓同一個 LAN 的筆電可以連進 Capture Hub。Windows Firewall 需要允許 TCP 8878。

若只使用同機 Agent，可維持：

```env
CAPTURE_HUB_HOST=127.0.0.1
```

## 遠端筆電啟動

開發版：

```powershell
$env:MOXUE_CAPTURE_HUB_URL="ws://192.168.1.20:8878/capture/ws"
python -m discord_ai_assistant.capture_agent.cli
```

或：

```powershell
python -m discord_ai_assistant.capture_agent.cli --hub-url "ws://192.168.1.20:8878/capture/ws"
```

Standalone EXE：

```powershell
MoxueCapture.exe --hub-url "ws://192.168.1.20:8878/capture/ws"
```

Agent 第一次連線會在 console 顯示 6 碼配對碼。配對 token 會永久保存；重新啟動後會自動驗證，不需要每次重新配對。

開發版資料位於 repo：

```text
data/capture-agent/
```

PyInstaller standalone EXE 則固定存在：

```text
%LOCALAPPDATA%\MoxueCapture\
├─ agent.json
├─ profiles.json
├─ remote.json
└─ calibration\
```

因此 `--onefile` 每次解壓到不同暫存目錄也不會遺失裝置 ID、ROI Profile 或配對 token。

## Discord 指令

```text
/meeting agents
/meeting pair <code>
/meeting select <agent_id>
/meeting agent
/meeting calibrate [game] [profile_name] [delay_seconds]
/meeting profiles
/meeting profile <profile_id>
/meeting ocr
```

### 第一次遠端配對

1. 在筆電啟動 `MoxueCapture`，並指定 Server 的 Hub URL。
2. 筆電 log 顯示：

```text
Capture Agent pairing code: A7K4X2
```

3. Discord：

```text
/meeting agents
```

會看到待配對裝置與配對碼。

4. DJ／管理員執行：

```text
/meeting pair code:A7K4X2
```

5. 該 Agent 會自動成為此 Discord Server 目前選取的 Capture Agent。

之後可用：

```text
/meeting select agent_id:<id>
```

在多台已配對 Agent 間切換。

### `/meeting calibrate`

若目前選取的遠端 Agent 在線，校準會發送到遠端裝置；否則 fallback 到 Server 本機 Agent。

1. Discord 顯示目標裝置與倒數。
2. 使用者在目標裝置切回全螢幕遊戲並讓聊天室顯示。
3. Agent 擷取一張凍結畫面。
4. GUI 在**目標遊戲裝置**以最上層無邊框視窗顯示凍結畫面。
5. 左鍵拖曳框選聊天室。
6. `Enter` 儲存、`Esc` 取消。
7. ROI 同時保存 normalized coordinates 與目前 pixel coordinates。

這個流程不需要把透明 GUI 持續疊在遊戲 live render 上，因此不需要 DirectX hook 或注入。

## 裝置識別與 Profiles

第一次啟動每台 Capture Agent 時會建立隨機 UUID 與本機顯示名稱，不使用硬碟序號、主機板序號或其他硬體指紋。

每個 Profile 保存遊戲名稱、校準時螢幕尺寸與 normalized ROI。解析度改變後會重新換算目前 pixel ROI。

Server 的 Agent 配對與每個 Discord Guild 目前選取的 Agent 存在：

```text
data/capture-hub/state.json
```

以上 Server runtime data 皆位於 Git 忽略的 `data/`。

## Capture backend

Agent 優先使用 DXcam / Desktop Duplication；失敗時 fallback 到 Pillow `ImageGrab`。Capture Agent 與 Bot/Meeting Audio 分成不同 process，避免 DXcam/comtypes 與 SoundCard/comtypes 的 Windows COM apartment 初始化互相衝突。

## 打包成不需要 Python 的 EXE

開發者電腦：

```powershell
python -m pip install -e ".[meeting,capture-build]"
.\scripts\build_capture_agent.ps1
```

輸出：

```text
dist\MoxueCapture.exe
```

也可：

```powershell
.\scripts\build_capture_agent.ps1 -InstallBuildTools
```

使用者端不需要另外安裝 Python、pip 或 venv。

## 下一階段：遠端 Session Capture

目前遠端 RPC 已涵蓋：

- Agent 狀態
- Profile 列表／切換
- 全螢幕 GUI 校準
- OCR 測試

下一階段會把 `SoundCard loopback + Whisper + Session audio/OCR event stream` 搬到被選中的 Agent，讓 Server 只負責 Session orchestration、Gemini 摘要與 Discord 發布。
