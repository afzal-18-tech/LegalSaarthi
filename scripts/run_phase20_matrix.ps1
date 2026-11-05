# Phase 20 main experimental matrix (E1-E8). Sequential; each run writes its own
# experiments/phase20/<run_id>/ with raw predictions. Skips experiments already
# marked done in experiments/phase20/matrix_progress.txt so it is resumable.
# Frozen: Qwen3-8B generator+judge, temperature 0, k=5, tau*=0.5 (Phase 17), no fallback.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force experiments/phase20 | Out-Null
$prog = "experiments/phase20/matrix_progress.txt"
if (-not (Test-Path $prog)) { New-Item -ItemType File $prog | Out-Null }

$common = @("--index-root", "experiments/phase19", "--phase-dir", "phase20", "--phase-num", "20")
$exps = [ordered]@{
  "E1" = @("-m","scripts.run_generation_eval","--system","generic") + $common
  "E2" = @("-m","scripts.run_generation_eval","--system","rag","--retriever","dense") + $common
  "E3" = @("-m","scripts.run_generation_eval","--system","rag","--retriever","bm25") + $common
  "E4" = @("-m","scripts.run_generation_eval","--system","rag","--retriever","hybrid") + $common
  "E5" = @("-m","scripts.run_generation_eval","--system","verified-rag","--retriever","bm25") + $common
  "E6" = @("-m","scripts.run_generation_eval","--system","verified-rag","--retriever","hybrid") + $common
  "E7" = @("-m","scripts.run_summarization_eval","--split","test","--final","--systems","generic_llm,structured_llm",
           "--restrict-manifest","data/metadata/split_manifest_prototype.csv","--phase-dir","phase20","--phase-num","20")
  "E8" = @("-m","scripts.run_summarization_eval","--split","test","--final","--systems","plain_llm",
           "--restrict-manifest","data/metadata/split_manifest_prototype.csv","--phase-dir","phase20","--phase-num","20")
}
foreach ($k in $exps.Keys) {
  if ((Get-Content $prog) -contains $k) { Write-Host "skip $k (done)"; continue }
  Write-Host "=== $k START $(Get-Date -Format o)"
  & python @($exps[$k])
  if ($LASTEXITCODE -eq 0) { Add-Content $prog $k; Write-Host "=== $k DONE $(Get-Date -Format o)" }
  else { Write-Host "=== $k FAILED exit=$LASTEXITCODE"; break }
}
