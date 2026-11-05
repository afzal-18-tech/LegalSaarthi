# Phase 21 ablation runs that need new LLM calls (everything else is derived offline
# from Phase 20 predictions). Resumable via experiments/phase21/matrix_progress.txt.
#   A1: Verified-RAG, DENSE retriever (completes the {Basic,Verified} x {dense,bm25,hybrid} grid)
#   S1: Summarization input-vs-format decoupling on the ORIGINAL 29 VALIDATION cases
#       (validation, not the locked test set: ablations must not add extra test-set looks)
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force experiments/phase21 | Out-Null
$prog = "experiments/phase21/matrix_progress.txt"
if (-not (Test-Path $prog)) { New-Item -ItemType File $prog | Out-Null }

$exps = [ordered]@{
  "A1" = @("-m","scripts.run_generation_eval","--system","verified-rag","--retriever","dense",
           "--index-root","experiments/phase19","--phase-dir","phase21","--phase-num","21")
  "S1" = @("-m","scripts.run_summarization_eval","--split","val",
           "--systems","generic_llm,structured_llm,structured_input_free_prompt,lead_input_structured_prompt",
           "--restrict-manifest","data/metadata/split_manifest_prototype.csv",
           "--phase-dir","phase21","--phase-num","21")
}
foreach ($k in $exps.Keys) {
  if ((Get-Content $prog) -contains $k) { Write-Host "skip $k (done)"; continue }
  Write-Host "=== $k START $(Get-Date -Format o)"
  & python @($exps[$k])
  if ($LASTEXITCODE -eq 0) { Add-Content $prog $k; Write-Host "=== $k DONE $(Get-Date -Format o)" }
  else { Write-Host "=== $k FAILED exit=$LASTEXITCODE"; break }
}
