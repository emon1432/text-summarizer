/**
 * AI Powered Text Summarization - Frontend Interactive client mechanics.
 * Handles live input token analysis, asynchronous form loading UI animations,
 * sample literature injection, clipboard utilities, and JSON analytical exporting.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Element Reference Selection
    const textarea = document.getElementById("input-text");
    const wordCountDisplay = document.getElementById("word-count");
    const charCountDisplay = document.getElementById("char-count");
    const clearBtn = document.getElementById("clear-btn");
    const sampleBtn = document.getElementById("sample-btn");
    const form = document.getElementById("summarize-form");
    const loadingOverlay = document.getElementById("loading-overlay");
    const loadingStatusText = document.getElementById("loading-status-text");

    // Fix Browser Back Button (bfcache) Lockout Feature
    window.addEventListener("pageshow", () => {
        if (loadingOverlay) {
            loadingOverlay.classList.remove("d-flex");
            loadingOverlay.classList.add("d-none");
        }
        const submitBtn = document.getElementById("submit-btn");
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.style.opacity = "1";
            submitBtn.style.pointerEvents = "auto";
        }
        if (textarea) textarea.readOnly = false;
    });



    // Real-time Word & Character Calculation Engine
    const updateCounts = () => {
        if (!textarea || !wordCountDisplay || !charCountDisplay) return;

        const text = textarea.value;
        const charCount = text.length;
        
        // Split across whitespace and filter empty strings for exact word counting
        const trimmed = text.trim();
        const words = trimmed ? trimmed.split(/\s+/).length : 0;

        charCountDisplay.textContent = charCount.toLocaleString();
        wordCountDisplay.textContent = words.toLocaleString();

        // Visual feedback for threshold compliance (consistent with 10,000 max words in backend)
        if (words > 10000) {
            wordCountDisplay.classList.remove("text-light");
            wordCountDisplay.classList.add("text-danger");
        } else {
            wordCountDisplay.classList.remove("text-danger");
            wordCountDisplay.classList.add("text-light");
        }
    };

    if (textarea) {
        textarea.addEventListener("input", updateCounts);
        // Initialize count display upon reload or back-navigation
        updateCounts();
    }

    // Clear Buffer Terminal Event Handler
    if (clearBtn && textarea) {
        clearBtn.addEventListener("click", () => {
            textarea.value = "";
            textarea.focus();
            updateCounts();
        });
    }

    // Sample Academic Literature Auto-Loader
    if (sampleBtn && textarea) {
        const sampleLiterature = `Artificial intelligence and deep natural language processing have undergone transformative paradigm shifts with the advent of attention-based sequence-to-sequence transformer models. Historically, Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) architectures suffered from gradient vanishing challenges when tasked with evaluating lengthy documents, causing significant context degradation across sequential timestamps.

To overcome these structural bottlenecks, bidirectional transformers such as BERT revolutionized contextual representation modeling through self-attention mechanics and masked language modeling objectives. Simultaneously, autoregressive models like OpenAI's GPT introduced exceptional generative text capabilities by predicting sequential tokens in a left-to-right decoding framework. 

In response to the distinct structural advantages of both encoding and decoding methodologies, researchers at Meta AI formulated BART (Bidirectional and Auto-Regressive Transformers). BART conceptually unifies a bidirectional BERT encoder with an autoregressive GPT decoder. During pre-training, input documents are purposely degraded by applying arbitrary noising transformations—including sentence shuffling, text intrenching, and continuous span masking. The neural decoder is subsequently tasked with reconstructing the original pristine text via cross-attention mechanisms over the encoder's representations.

When specifically fine-tuned on comprehensive journalistic repositories such as the CNN/DailyMail abstractive summarization dataset, BART demonstrates remarkable cognitive abstraction. Unlike extractive summarizers that simply rank and extract isolated verbatim sentences, abstractive transformer architectures actively synthesize information, condense complex clauses, and construct entirely new grammatical compositions that faithfully preserve fundamental semantic propositions while drastically diminishing overall data volume.`;

        sampleBtn.addEventListener("click", () => {
            textarea.value = sampleLiterature.trim();
            updateCounts();
            textarea.focus();
            
            // Animate sample button visual feedback
            const originalHTML = sampleBtn.innerHTML;
            sampleBtn.innerHTML = `<i class="fa-solid fa-check text-success"></i> <span>Article Loaded!</span>`;
            setTimeout(() => { sampleBtn.innerHTML = originalHTML; }, 2000);
        });
    }

    // Form Submission Interception & Dynamic Loading Animation
    if (form && loadingOverlay) {
        form.addEventListener("submit", (event) => {
            const trimmedText = textarea ? textarea.value.trim() : "";
            const words = trimmedText ? trimmedText.split(/\s+/).length : 0;
            
            // Consistent frontend/backend validation constraints (5 to 10,000 words)
            if (words < 5) {
                event.preventDefault();
                alert("Input text is too abbreviated (< 5 words). Please provide a fuller sentence or paragraph before generating a summary.");
                if (textarea) textarea.focus();
                return false;
            }
            
            if (words > 10000) {
                event.preventDefault();
                alert(`Input length (${words.toLocaleString()} words) exceeds the maximum threshold of 10,000 words. Please shorten your text before submitting.`);
                if (textarea) textarea.focus();
                return false;
            }

            // Reveal high-immersion loading overlay
            loadingOverlay.classList.remove("d-none");
            loadingOverlay.classList.add("d-flex");
            
            // Lock input controls to prevent double-submission bugs
            const submitBtn = document.getElementById("submit-btn");
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.style.opacity = "0.75";
                submitBtn.style.pointerEvents = "none";
            }
            if (textarea) textarea.readOnly = true;

            // Rotate simulation messages during forward neural passes
            const statusMessages = [
                "Tokenizing text input via BPE SentencePiece vocabulary...",
                "Routing neural tensor representations onto computation accelerator...",
                "Evaluating cross-attention matrices via deep Beam Search exploration...",
                "Distilling abstractive semantic summary sequences...",
                "Finalizing compression statistics and runtime analytics..."
            ];
            let msgIndex = 0;
            const messageInterval = setInterval(() => {
                if (loadingStatusText && msgIndex < statusMessages.length) {
                    loadingStatusText.textContent = statusMessages[msgIndex];
                    msgIndex++;
                } else {
                    clearInterval(messageInterval);
                }
            }, 1800);
        });
    }

    // Clipboard Copy Mechanics for Result Viewports
    const setupCopyButtons = () => {
        const copyButtons = document.querySelectorAll(".copy-btn");
        copyButtons.forEach((btn) => {
            btn.addEventListener("click", async () => {
                const targetSelector = btn.getAttribute("data-clipboard-target");
                const targetContainer = document.querySelector(targetSelector);
                
                if (targetContainer && navigator.clipboard) {
                    const textToCopy = targetContainer.innerText || targetContainer.textContent;
                    try {
                        await navigator.clipboard.writeText(textToCopy.trim());
                        const origContent = btn.innerHTML;
                        btn.innerHTML = `<i class="fa-solid fa-check text-success"></i> <span>Copied! ✓</span>`;
                        btn.classList.add("border-success", "text-success");
                        
                        setTimeout(() => {
                            btn.innerHTML = origContent;
                            btn.classList.remove("border-success", "text-success");
                        }, 2200);
                    } catch (err) {
                        console.error("Clipboard copy failure:", err);
                        alert("Could not automatically copy text to clipboard.");
                    }
                }
            });
        });
    };
    setupCopyButtons();

    // Academic Metrics JSON Export Handler
    const exportJsonBtn = document.getElementById("export-json-btn");
    const summaryDataEl = document.getElementById("summary-data-json");

    if (exportJsonBtn && summaryDataEl) {
        exportJsonBtn.addEventListener("click", () => {
            try {
                const jsonData = JSON.parse(summaryDataEl.textContent);
                const jsonString = JSON.stringify(jsonData, null, 2);
                const blob = new Blob([jsonString], { type: "application/json" });
                const downloadUrl = URL.createObjectURL(blob);
                
                const downloadLink = document.createElement("a");
                downloadLink.href = downloadUrl;
                const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, "");
                downloadLink.download = `summarization_report_${timestamp}.json`;
                
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
                URL.revokeObjectURL(downloadUrl);

                // Feedback animation
                const origText = exportJsonBtn.innerHTML;
                exportJsonBtn.innerHTML = `<i class="fa-solid fa-check text-success"></i> <span>Report Downloaded!</span>`;
                setTimeout(() => { exportJsonBtn.innerHTML = origText; }, 2500);
            } catch (ex) {
                console.error("JSON export exception occurred:", ex);
                alert("An anomaly prevented compiling the JSON analytics report.");
            }
        });
    }
});
