# Document Generation Tool

當使用者明確要求下載式文件、Word、Markdown、PowerPoint、簡報或口說稿時使用。

## create_presentation_package

從結構化 slide 資料建立兩個檔案：

- .pptx：有固定版型、標題頁、段落層次、頁碼與主題樣式的 PowerPoint。
- .docx：依投影片順序整理的口說稿，每頁含投影片標題、重點與 speaker notes。

請把投影片畫面保持精簡；詳細講法放在 speaker_notes。一般建議 5–12 張，不要把長篇文章直接塞進每張投影片。

theme 可用：

- modern：深色標題區、醒目 accent，適合一般報告。
- academic：較正式、適合課堂／研究報告。
- minimal：乾淨留白、低裝飾。

## create_document

建立內容相同的 .docx 與 .md。每個 section 可有 heading、body、bullets。

## Artifact boundary

所有輸出只寫入 data/tool-artifacts/document_generation/，並透過 _moxue_artifacts 交回 Core。Core 會再次驗證路徑、大小、MIME 與 Office ZIP 結構後才允許 Discord 發送。

此工具不使用 Canva、Google Drive 或其他外部文件服務，不需要額外 OAuth。
