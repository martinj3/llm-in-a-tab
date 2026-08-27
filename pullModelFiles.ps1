#requires -Version 7
$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'   # otherwise IWR's progress bar dominates runtime

$models = [ordered]@{
    '135M' = 'HuggingFaceTB/SmolLM2-135M-Instruct'
    '360M' = 'HuggingFaceTB/SmolLM2-360M-Instruct'
}
$files = @(
    'config.json'
    'generation_config.json'
    'tokenizer_config.json'
    'special_tokens_map.json'
    'tokenizer.json'      # big (few MB) - the full vocab + merges
    'vocab.json'          # may not exist
    'merges.txt'          # may not exist
)

$results = foreach ($tag in $models.Keys) {
    $repo = $models[$tag]
    $dir  = Join-Path 'reference' $tag
    New-Item -ItemType Directory -Path $dir -Force | Out-Null

    foreach ($f in $files) {
        $url = "https://huggingface.co/$repo/resolve/main/$f"
        $out = Join-Path $dir $f
        try {
            Invoke-WebRequest -Uri $url -OutFile $out -ErrorAction Stop
            [pscustomobject]@{ Model = $tag; File = $f; Status = 'ok'; Bytes = (Get-Item $out).Length }
        } catch {
            if (Test-Path $out) { Remove-Item $out -Force }
            $code = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
            [pscustomobject]@{ Model = $tag; File = $f; Status = "MISSING ($code)"; Bytes = 0 }
        }
    }
}

$results | Format-Table -AutoSize

# --- immediate sanity check: does the real config match what we've been assuming? ---
Write-Host "`nConfig summary:" -ForegroundColor Cyan
$summary = foreach ($tag in $models.Keys) {
    $p = "reference/$tag/config.json"
    if (-not (Test-Path $p)) { continue }
    $c = Get-Content $p -Raw | ConvertFrom-Json
    [pscustomobject]@{
        Model  = $tag
        hidden = $c.hidden_size
        layers = $c.num_hidden_layers
        heads  = $c.num_attention_heads
        kv     = $c.num_key_value_heads
        ffn    = $c.intermediate_size
        vocab  = $c.vocab_size
        rope   = $c.rope_theta
        tied   = $c.tie_word_embeddings
        act    = $c.hidden_act
    }
}
$summary | Format-Table -AutoSize