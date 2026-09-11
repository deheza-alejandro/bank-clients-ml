param([string]$commitMsgFile)

if (Test-Path $commitMsgFile) {
    $content = [System.IO.File]::ReadAllText($commitMsgFile)
    
    $parts = $content -split '(\r?\n\r?\n)', 2

    if ($parts.Count -eq 3) {
        $titleAndBlankLine = $parts[0] + $parts[1]
        $body = $parts[2]
        
        $formattedBody = $body -replace '(.{1,71})(?:\s+|$)', ('$1' + [char]10)
        
        [System.IO.File]::WriteAllText($commitMsgFile, ($titleAndBlankLine + $formattedBody))
    }
}
