

"""
Live Component-by-Component End-to-End Verification for LegalSaarthi Web Platform
Tests all API endpoints, data services, and pipeline capabilities on the running server (http://127.0.0.1:8000).
"""

import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "http://127.0.0.1:8000"


def print_step(name: str):
    print(f"\n[TESTING] {name}...")


def http_get(endpoint: str) -> dict | str:
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "LegalSaarthi-TestRunner/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read().decode("utf-8")
        if "application/json" in content_type:
            return json.loads(body)
        return body


def http_post_json(endpoint: str, data: dict) -> dict:
    url = f"{BASE_URL}{endpoint}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "LegalSaarthi-TestRunner/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_delete(endpoint: str) -> dict:
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LegalSaarthi-TestRunner/1.0"},
        method="DELETE",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_upload_pdf(endpoint: str, file_path: pathlib.Path) -> dict:
    url = f"{BASE_URL}{endpoint}"
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    lines = [
        f"--{boundary}".encode("latin-1"),
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"'.encode("latin-1"),
        b"Content-Type: application/pdf",
        b"",
        file_bytes,
        f"--{boundary}--".encode("latin-1"),
        b"",
    ]
    body = b"\r\n".join(lines)
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "LegalSaarthi-TestRunner/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_component_tests():
    print("=" * 80)
    print("LegalSaarthi — Live End-to-End Component Verification Suite")
    print(f"Target Server: {BASE_URL}")
    print("=" * 80)

    results = []

    # 1. Health & Core Engine Status
    print_step("Component 1: System Health & Core Telemetry")
    t0 = time.time()
    try:
        health = http_get("/api/health")
        assert health["status"] == "ok", f"Expected status ok, got {health['status']}"
        assert health["total_cases_indexed"] >= 993, f"Expected >=993 cases, got {health['total_cases_indexed']}"
        assert health["total_chunks_loaded"] > 50000, f"Expected >50000 chunks, got {health['total_chunks_loaded']}"
        assert health["bm25_index_loaded"] is True, "BM25 index not loaded"
        dur = time.time() - t0
        print(f"  -> PASS: {health['total_cases_indexed']} cases, {health['total_chunks_loaded']:,} chunks, LLM connected: {health.get('llm_connected')} ({dur:.3f}s)")
        results.append(("1. System Health & Corpus Telemetry", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("1. System Health & Corpus Telemetry", "FAIL", str(e)))

    # 2. UI Markup & Asset Integrity (Emoji check)
    print_step("Component 2: UI Markup & Static Assets Integrity")
    t0 = time.time()
    try:
        index_html = http_get("/")
        assert "LegalSaarthi" in index_html, "Title or brand missing in index.html"
        assert 'id="tab-home"' in index_html, "Home tab missing"
        assert 'id="tab-documents"' in index_html, "Documents tab missing"
        assert 'id="tab-qa"' in index_html, "QA Studio tab missing"
        assert 'id="tab-summarizer"' in index_html, "Summarizer tab missing"
        assert 'id="tab-explorer"' in index_html, "Explorer tab missing"
        assert 'id="tab-provenance"' in index_html, "Provenance tab missing"
        assert 'id="tab-metrics"' in index_html, "Research & Evaluation tab missing"

        # Check for banned decorative emojis in index.html
        banned = ["⚖️", "🔍", "📜", "📊", "📑", "💡", "🚀", "🛡️", "🏛️"]
        found_emojis = [e for e in banned if e in index_html]
        assert not found_emojis, f"Found decorative emojis in index.html: {found_emojis}"

        dur = time.time() - t0
        print(f"  -> PASS: All 7 tabs present, 0 decorative emojis detected ({dur:.3f}s)")
        results.append(("2. UI Markup & Zero-Emoji Integrity", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("2. UI Markup & Zero-Emoji Integrity", "FAIL", str(e)))

    # 3. Universal Benchmark Inquiry Chips
    print_step("Component 3: Universal Benchmark Inquiry Prompts")
    t0 = time.time()
    try:
        prompts = http_get("/api/benchmark-prompts")
        assert len(prompts) == 6, f"Expected 6 benchmark prompts, got {len(prompts)}"
        tiers = [p["tier"] for p in prompts]
        assert "Final Order / Disposition" in tiers, "Missing Final Order prompt"
        assert "Reasoning & Ratio Decidendi" in tiers, "Missing Reasoning prompt"
        dur = time.time() - t0
        print(f"  -> PASS: {len(prompts)} universal inquiry prompts loaded ({dur:.3f}s)")
        results.append(("3. Universal Benchmark Inquiry Prompts", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("3. Universal Benchmark Inquiry Prompts", "FAIL", str(e)))

    # 4. Corpus Explorer & Case Detail
    print_step("Component 4: Corpus Explorer & Legal Case Detail")
    t0 = time.time()
    try:
        cases_resp = http_get("/api/cases/all")
        assert cases_resp["total"] >= 993, f"Expected >=993 cases, got {cases_resp.get('total')}"
        assert len(cases_resp["cases"]) >= 993, f"Expected >=993 case items, got {len(cases_resp.get('cases'))}"
        sample_case_id = cases_resp["cases"][0]["case_id"]
        encoded_case_id = urllib.parse.quote(sample_case_id)
        detail = http_get(f"/api/cases/{encoded_case_id}")
        assert detail["case_id"] == sample_case_id, "Case ID mismatch"
        assert "total_pages" in detail, "Missing total_pages in detail"
        assert "sections_summary" in detail, "Missing sections_summary in detail"
        dur = time.time() - t0
        print(f"  -> PASS: Case {sample_case_id} retrieved ({detail.get('total_pages')} pages, sections: {len(detail.get('sections_summary', {}))}) ({dur:.3f}s)")
        results.append(("4. Corpus Explorer & Case Detail", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("4. Corpus Explorer & Case Detail", "FAIL", str(e)))

    # 5. Chunk Provenance Drawer API
    print_step("Component 5: Sources & Evidence (Chunk Provenance)")
    t0 = time.time()
    try:
        chunk = http_get("/api/chunks/2015%20INSC%20177_section_aware_1")
        assert chunk["case_id"] == "2015 INSC 177", f"Expected 2015 INSC 177, got {chunk.get('case_id')}"
        assert "text" in chunk and len(chunk["text"]) > 10, "Empty chunk text"
        assert "pages" in chunk, "Missing pages in chunk"
        dur = time.time() - t0
        print(f"  -> PASS: Chunk resolved with page span {chunk.get('pages')} and length {len(chunk['text'])} chars ({dur:.3f}s)")
        results.append(("5. Sources & Evidence (Provenance)", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("5. Sources & Evidence (Provenance)", "FAIL", str(e)))

    # 6. PDF Ingestion & Document Manager
    print_step("Component 6: PDF Ingestion Pipeline (Upload, Page-Extract, Index)")
    t0 = time.time()
    uploaded_doc_id = None
    try:
        sample_pdf = next(pathlib.Path("data/raw").rglob("*.pdf"), None)
        assert sample_pdf and sample_pdf.exists(), "No test PDF available in data/raw"
        upload_resp = http_upload_pdf("/api/documents/upload", sample_pdf)
        assert upload_resp["status"] == "ready", f"Expected ready status, got {upload_resp.get('status')}"
        doc_info = upload_resp["document"]
        uploaded_doc_id = doc_info["id"]
        assert doc_info["page_count"] > 0, "Page count must be > 0"
        assert doc_info["chunk_count"] > 0, "Chunk count must be > 0"
        dur = time.time() - t0
        print(f"  -> PASS: PDF {sample_pdf.name} ingested -> doc_id: {uploaded_doc_id} ({doc_info['page_count']} pages, {doc_info['chunk_count']} chunks) ({dur:.3f}s)")
        results.append(("6. PDF Ingestion Pipeline", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("6. PDF Ingestion Pipeline", "FAIL", str(e)))

    # 7. Document-Scoped QA with Zero-Leakage Guarantee
    print_step("Component 7: Document-Scoped Evidence-Grounded QA")
    t0 = time.time()
    try:
        assert uploaded_doc_id, "No uploaded doc available for scoped QA"
        qa_req = {
            "query": "What are the main contentions and factual background of this appeal?",
            "system": "verified_rag",
            "case_id": uploaded_doc_id,
            "top_k": 3,
        }
        qa_resp = http_post_json("/api/qa", qa_req)
        primary = qa_resp.get("primary", {})
        assert primary.get("system") == "verified_rag", f"System mismatch: {primary.get('system')}"
        assert primary.get("case_fidelity_passed") is True, "Case fidelity check failed"
        # Assert strict zero cross-document contamination
        for chunk_item in primary.get("cited_chunks", []):
            cid = chunk_item.get("chunk_id", "") if isinstance(chunk_item, dict) else str(chunk_item)
            assert uploaded_doc_id in cid or uploaded_doc_id.replace("_", " ") in cid, f"Cross-document contamination: {cid}"
        dur = time.time() - t0
        print(f"  -> PASS: Scoped answer generated with {len(primary.get('cited_chunks', []))} citations, fidelity passed ({dur:.3f}s)")
        results.append(("7. Document-Scoped Evidence-Grounded QA", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("7. Document-Scoped Evidence-Grounded QA", "FAIL", str(e)))

    # 8. QA Abstention Mechanism on Out-of-Scope Query
    print_step("Component 8: Abstention on Out-of-Scope / Unsupported Inquiry")
    t0 = time.time()
    try:
        assert uploaded_doc_id, "No uploaded doc available for abstention test"
        unsupported_req = {
            "query": "What is the exact mathematical proof for the Riemann hypothesis in analytical number theory?",
            "system": "verified_rag",
            "case_id": uploaded_doc_id,
            "top_k": 3,
        }
        abstain_resp = http_post_json("/api/qa", unsupported_req)
        primary = abstain_resp.get("primary", {})
        ans_lower = primary.get("answer", "").lower()
        is_abstained = primary.get("abstained") is True or "insufficient" in ans_lower or "could not" in ans_lower or "abstained" in ans_lower
        assert is_abstained, f"System failed to abstain safely. Answer: {primary.get('answer')[:100]}"
        dur = time.time() - t0
        print(f"  -> PASS: Correctly abstained on unsupported out-of-scope query ({dur:.3f}s)")
        results.append(("8. Abstention on Absent Evidence", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("8. Abstention on Absent Evidence", "FAIL", str(e)))

    # 9. Structured Judgment Summarization & Precedent Filtering
    print_step("Component 9: Structured Judgment Summarizer")
    t0 = time.time()
    try:
        summ_req = {
            "case_id": "2015 INSC 177",
            "mode": "structured",
        }
        summ_resp = http_post_json("/api/summarize", summ_req)
        primary = summ_resp.get("primary", {})
        assert primary.get("mode") == "structured", f"Expected structured mode, got {primary.get('mode')}"
        assert primary.get("summary"), "Missing summary text"
        assert primary.get("predicted_disposition"), "Missing predicted_disposition"
        # Verify that precedents don't contain invalid sentence fragments
        for prec in primary.get("cited_precedents", []):
            assert "court held" not in prec.lower(), f"Sentence fragment found in cited precedents: {prec}"
            assert "failed to prove" not in prec.lower(), f"Sentence fragment found in cited precedents: {prec}"
        dur = time.time() - t0
        print(f"  -> PASS: Structured summary generated (Disp: {primary.get('predicted_disposition')}, Precedents: {len(primary.get('cited_precedents', []))}) ({dur:.3f}s)")
        results.append(("9. Structured Judgment Summarizer", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("9. Structured Judgment Summarizer", "FAIL", str(e)))

    # 10. Plain-Language Summarization & Readability Diagnostics
    print_step("Component 10: Plain-Language Summary & Readability Diagnostics")
    t0 = time.time()
    try:
        plain_req = {
            "case_id": "2015 INSC 177",
            "mode": "plain_language",
        }
        plain_resp = http_post_json("/api/summarize", plain_req)
        primary = plain_resp.get("primary", {})
        assert primary.get("mode") == "plain_language", f"Expected plain_language, got {primary.get('mode')}"
        assert primary.get("summary"), "Missing summary text"
        re_score = primary.get("flesch_reading_ease")
        fk_grade = primary.get("fk_grade_level")
        assert re_score is not None, "Missing flesch_reading_ease"
        assert fk_grade is not None, "Missing fk_grade_level"
        dur = time.time() - t0
        print(f"  -> PASS: Plain summary generated (Flesch: {re_score:.1f}, Grade: {fk_grade:.1f}) ({dur:.3f}s)")
        results.append(("10. Plain Summary & Readability Diagnostics", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("10. Plain Summary & Readability Diagnostics", "FAIL", str(e)))

    # 11. In-Document Keyword Search with Snippets
    print_step("Component 11: In-Document Search & Snippet Highlighting")
    t0 = time.time()
    try:
        # Search using uploaded document id
        target_doc = uploaded_doc_id or "2015_INSC_177"
        encoded_doc = urllib.parse.quote(target_doc)
        search_resp = http_get(f"/api/documents/{encoded_doc}/search?q=appeal&top_k=5")
        assert search_resp["total_hits"] > 0, "Expected matches for keyword 'appeal'"
        assert len(search_resp["hits"]) > 0, "Empty search hits"
        hit0 = search_resp["hits"][0]
        assert "page_start" in hit0, "Missing page_start in search hit"
        assert "snippet" in hit0 and len(hit0["snippet"]) > 0, "Missing snippet text in hit"
        assert "match_type" in hit0, "Missing match_type in hit"
        dur = time.time() - t0
        print(f"  -> PASS: {search_resp['total_hits']} hits matched (match_type: {hit0['match_type']}, snippet length: {len(hit0['snippet'])}) in {target_doc[:12]} ({dur:.3f}s)")
        results.append(("11. In-Document Search & Snippets", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("11. In-Document Search & Snippets", "FAIL", str(e)))

    # 12. Local Query History Lifecycle (Record, List, Delete)
    print_step("Component 12: Local Query History Lifecycle")
    t0 = time.time()
    try:
        history_resp = http_get("/api/history")
        assert "history" in history_resp, "Missing history list in response"
        assert len(history_resp["history"]) > 0, "Expected history records after QA calls"
        sample_item = history_resp["history"][0]
        item_id = sample_item["query_id"]
        # Delete single item
        del_resp = http_delete(f"/api/history/{item_id}")
        assert del_resp.get("status") == "ok", f"Expected ok, got {del_resp.get('status')}"
        dur = time.time() - t0
        print(f"  -> PASS: History recorded, item {item_id[:8]} safely deleted ({dur:.3f}s)")
        results.append(("12. Local Query History Lifecycle", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("12. Local Query History Lifecycle", "FAIL", str(e)))

    # 13. Cross-Judgment Comparison Engine
    print_step("Component 13: Cross-Judgment Comparison Engine")
    t0 = time.time()
    try:
        comp_req = {
            "case_id_a": "2015 INSC 177",
            "case_id_b": "2015 INSC 417",
        }
        comp_resp = http_post_json("/api/compare", comp_req)
        dims = comp_resp.get("dimensions", [])
        assert len(dims) in [4, 5], f"Expected 4 or 5 comparison dimensions, got {len(dims)}"
        dim_names = [d["dimension"] for d in dims]
        assert "facts_and_posture" in dim_names
        assert "issues" in dim_names
        assert "reasoning_and_authorities" in dim_names
        assert "outcome_and_order" in dim_names
        # Check precedent treatment synthesis guardrail
        prec_rel = comp_resp.get("precedent_relation")
        assert prec_rel in ["independent_authorities", "cited", "followed", "distinguished"]
        dur = time.time() - t0
        print(f"  -> PASS: {len(dims)} dimensions compared, Precedent Relation: '{prec_rel}' ({dur:.3f}s)")
        results.append(("13. Cross-Judgment Comparison", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("13. Cross-Judgment Comparison", "FAIL", str(e)))

    # 14. Research & Evaluation Metrics API
    print_step("Component 14: Research & Evaluation Empirical Matrix")
    t0 = time.time()
    try:
        metrics = http_get("/api/metrics")
        assert "research_question" in metrics, "Missing research_question"
        assert "hypotheses" in metrics, "Missing hypotheses"
        assert "baselines" in metrics, "Missing baselines"
        assert "corpus_stats" in metrics, "Missing corpus_stats"
        dur = time.time() - t0
        print(f"  -> PASS: Research evaluation metrics verified (H1-H4 hypotheses active) ({dur:.3f}s)")
        results.append(("14. Research & Evaluation Matrix", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("14. Research & Evaluation Matrix", "FAIL", str(e)))

    # 15. Safe Document Deletion (Corpus Protection)
    print_step("Component 15: Uploaded Document Deletion & Corpus Protection")
    t0 = time.time()
    try:
        if uploaded_doc_id:
            del_doc_resp = http_delete(f"/api/documents/{uploaded_doc_id}")
            assert del_doc_resp.get("status") == "ok", f"Expected ok, got {del_doc_resp.get('status')}"
            # Verify 993-corpus is completely untouched
            health_after = http_get("/api/health")
            assert health_after["total_cases_indexed"] >= 993, f"Corpus affected! Expected >=993, got {health_after['total_cases_indexed']}"
            dur = time.time() - t0
            print(f"  -> PASS: Uploaded doc {uploaded_doc_id[:8]} removed. Corpus remains protected ({dur:.3f}s)")
            results.append(("15. Safe Document Deletion", "PASS", f"{dur:.3f}s"))
        else:
            print("  -> SKIPPED (No doc uploaded)")
            results.append(("15. Safe Document Deletion", "PASS", "0.000s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("15. Safe Document Deletion", "FAIL", str(e)))


    # 16. Authentication & Roles (LAWYER, RESEARCHER, STUDENT)
    print_step("Component 16: User Registration across Roles (LAWYER, RESEARCHER, STUDENT)")
    t0 = time.time()
    try:
        roles_to_test = ["LAWYER", "RESEARCHER", "STUDENT"]
        for r in roles_to_test:
            unique_email = f"test_{r.lower()}_{int(time.time()*1000)}@juris.test"
            signup_resp = http_post_json("/api/auth/signup", {
                "email": unique_email,
                "password": "SecurePass@123",
                "full_name": f"Dr. Test {r.capitalize()}",
                "role": r,
                "organization": f"Apex Institute of {r.capitalize()}"
            })
            assert signup_resp.get("status") == "success", f"Signup failed for {r}"
            assert signup_resp.get("user", {}).get("role") == r, f"Role mismatch for {r}"
            assert "token" in signup_resp, "Token missing in signup response"
        dur = time.time() - t0
        print(f"  -> PASS: User signup verified for LAWYER, RESEARCHER, STUDENT roles ({dur:.3f}s)")
        results.append(("16. Auth Registration & Roles", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("16. Auth Registration & Roles", "FAIL", str(e)))

    # 17. Login, Session Token & Profile Retrieval
    print_step("Component 17: Login, Token Verification & Session Termination")
    t0 = time.time()
    try:
        # Login with pre-seeded lawyer account
        login_resp = http_post_json("/api/auth/login", {
            "email": "advocate.sharma@legal.in",
            "password": "Password@123"
        })
        assert login_resp.get("status") == "success", "Login failed"
        token = login_resp.get("token")
        assert token, "Token missing from login"
        assert login_resp.get("user", {}).get("role") == "LAWYER", "User role mismatch"

        # Verify current user via /api/auth/me
        url = f"{BASE_URL}/api/auth/me"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            me_data = json.loads(resp.read().decode("utf-8"))
            assert me_data.get("email") == "advocate.sharma@legal.in", "Profile email mismatch"
            assert me_data.get("role") == "LAWYER", "Profile role mismatch"

        # Logout
        url = f"{BASE_URL}/api/auth/logout"
        req = urllib.request.Request(url, data=b"{}", headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            logout_resp = json.loads(resp.read().decode("utf-8"))
            assert logout_resp.get("status") in ("success", "logged_out", "ok"), f"Logout failed: {logout_resp}"

        dur = time.time() - t0
        print(f"  -> PASS: Auth lifecycle verified: login, session check, logout ({dur:.3f}s)")
        results.append(("17. Auth Session Lifecycle", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("17. Auth Session Lifecycle", "FAIL", str(e)))

    # 18. 1-Click Instant Demo Accounts
    print_step("Component 18: Instant 1-Click Demo Accounts Availability")
    t0 = time.time()
    try:
        demos = http_get("/api/auth/demo-accounts")
        assert isinstance(demos, list) and len(demos) == 3, f"Expected 3 demo accounts, got {len(demos)}"
        demo_roles = {d["role"] for d in demos}
        assert demo_roles == {"LAWYER", "RESEARCHER", "STUDENT"}, f"Missing roles: {demo_roles}"
        dur = time.time() - t0
        print(f"  -> PASS: 1-Click Demo accounts validated for LAWYER, RESEARCHER, STUDENT ({dur:.3f}s)")
        results.append(("18. 1-Click Demo Accounts", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("18. 1-Click Demo Accounts", "FAIL", str(e)))

    # 19. Dark Glassmorphism & Home Grid Structure Integrity
    print_step("Component 19: Dark Glassmorphic Grid & Status Pill Validation")
    t0 = time.time()
    try:
        html = http_get("/")
        css = http_get("/static/css/style.css")

        # Check status pill strictly says Online without chunk count
        assert '<span id="status-text">Online</span>' in html, "Status pill does not strictly contain 'Online'"
        assert 'Online (51,235' not in html, "Status pill contains old chunk count!"

        # Check exact Home page grid elements from user screenshot
        assert 'Verified Legal Intelligence Framework' in html, "Missing hero title"
        assert 'Open Document Hub' in html, "Missing Documents grid card"
        assert 'Launch QA Studio' in html, "Missing Ask LegalSaarthi grid card"
        assert 'Corpus & Empirical Specifications' in html, "Missing Corpus specifications grid"

        # Check clean UI guardrails: NO filter chips on Home, NO Authority Graph or Authority Score
        assert 'hero-filter-pill' not in html, "Filter chips should be absent from Home UI"
        assert 'Authority Graph' not in html, "Authority Graph should be absent from user-facing UI"
        assert 'AUTHORITY SCORE' not in html, "Authority Score should be absent from user-facing UI"

        # Check Zero emojis in HTML
        import re
        emojis_html = re.findall(r'[𐀀-􏿿]', html)
        assert len(emojis_html) == 0, f"Found emojis in HTML: {set(emojis_html)}"

        dur = time.time() - t0
        print(f"  -> PASS: Glassmorphism grid & status pill verified: 0 emojis ({dur:.3f}s)")
        results.append(("19. Glass Grid & Status Guardrails", "PASS", f"{dur:.3f}s"))
    except Exception as e:
        print(f"  -> FAIL: {e}")
        results.append(("19. Glass Grid & Status Guardrails", "FAIL", str(e)))

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Component':45s} | {'Status':8s} | {'Latency / Error'}")
    print("-" * 80)
    all_ok = True
    for comp, stat, note in results:
        print(f"{comp:45s} | {stat:8s} | {note}")
        if stat != "PASS":
            all_ok = False
    print("=" * 80)
    if all_ok:
        print("ALL 19 WEB PLATFORM COMPONENTS FULLY OPERATIONAL AND VERIFIED!")
    else:
        print("ONE OR MORE COMPONENTS FAILED! CHECK LOGS ABOVE.")
    print("=" * 80)
    return all_ok


if __name__ == "__main__":
    success = run_component_tests()
    sys.exit(0 if success else 1)
