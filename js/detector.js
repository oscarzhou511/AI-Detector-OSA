document.addEventListener('DOMContentLoaded', () => {
            const GOOGLE_CLIENT_ID = "810577671337-sctto8p3i003l9renmhhlv4fs2pg69ik.apps.googleusercontent.com"; // <-- PASTE YOUR CLIENT ID HERE
            const publicKey = `-----BEGIN PUBLIC KEY-----
MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEA2YlQJJaB9O3X91mYfk9z
3U0Zt4oC95TXVV8omGmCpeNc53Z392oXhkrWkzUsae90BWO2ORqtT9e/+OMW2I9e
zCUD6a82VwxxQXgt+uwgY1FfuZP3vARtdGrmiepcJ1XakiMhjs2Lw0BSs9BSCAo9
pOSvqV3osG97bug9BkacAPl/b1RxXcYoX0xnWN+WmwJtNQrd5twr8DS5MrdEX37l
J9MgD0m1Iby1sGwgvy0IFIb1bU2pbVp+jUoHrYX/dcmRb+VwgquuTyTsU1ktfqHX
AFxQpnTdhNC0thzWC4vc7PPCSQX6uFyJoH0fK+ezYcnhmOXX2J70KGUVV0KWW37c
UJl+xmTs/4hFLDQOUNoyIdOVmAkbmGawjpZPFSGtcy1R8sMgNm1aH+UlfvT7d5ky
xZ3mYjRebtIdcGp1ByqIRkHz+NIIO7UeKlBdGDHZxZl2gqA7teJyLRadF8xTnl84
yx5pLDdMjPPZZk8GQXtuO9lQAK2W/Xsks8qNdMvPUhEyn3JamiHOVJRJEERvsrEL
IZWSunkh8ysj6EW9vm5HzjZiTLdS+EtfNS6nAFkKSD2ythirxm9reDTvUK5pzTdM
mup25IPUoJIynx1AIhURGN9pFo96dTsjEGTuQDTGWsMdvk6WXQgQyVKEBelCrTvW
FQSY1UWJZzwwEm7JhHRz3MsCAwEAAQ==
-----END PUBLIC KEY-----`;

            const textInput = document.getElementById('text-input');
            const checkButton = document.getElementById('check-button');
            const resultsArea = document.getElementById('results-area');
            const loginView = document.getElementById('login-view');
            const appView = document.getElementById('app-view');
            const userEmailEl = document.getElementById('user-email');
            const userQuotaEl = document.getElementById('user-quota');
            const signOutButton = document.getElementById('signout-button');

            let idToken = null; // Store the user's ID token

            checkButton.addEventListener('click', handleAnalysis);
            signOutButton.addEventListener('click', handleSignOut);
            textInput.addEventListener('input', () => {
                const text = textInput.value.trim();
                document.getElementById('word-count').textContent = `${text.split(/\s+/).filter(Boolean).length} words`;
            });

            // This function is called by the Google GIS library after it loads
            window.onload = function () {
                google.accounts.id.initialize({
                    client_id: GOOGLE_CLIENT_ID,
                    callback: handleCredentialResponse
                });
                google.accounts.id.renderButton(
                    document.getElementById("google-signin-button"), {
                        theme: "outline",
                        size: "large"
                    }
                );
                google.accounts.id.prompt(); // Also display the One Tap prompt
            };

            function handleCredentialResponse(response) {
                // The response.credential is the ID token (a JWT)
                idToken = response.credential;

                // Decode the JWT to get user info without needing a library
                const payload = JSON.parse(atob(idToken.split('.')[1]));
                const userEmail = payload.email;

                userEmailEl.textContent = userEmail;

                if (userEmail.endsWith('@pembroke.sa.edu.au')) {
                    updateQuotaDisplay({ is_unlimited: true });
                    setTimeout(() => alert("Teachers and Staff from Pembroke School have unlimited access to this service."), 100);
                } else {
                    updateQuotaDisplay({ is_unlimited: false, words_used_today: 0, daily_limit: 1000 });
                }

                loginView.style.display = 'none';
                appView.style.display = 'block';
                resultsArea.classList.remove('visible');
            }

            function handleSignOut() {
                idToken = null;
                google.accounts.id.disableAutoSelect(); // Prevent auto-re-login
                loginView.style.display = 'block';
                appView.style.display = 'none';
            }

            // --- Helper functions for encryption/decryption are unchanged ---
            function arrayBufferToBase64(buffer) {
                let binary = '';
                const bytes = new Uint8Array(buffer);
                const len = bytes.byteLength;
                for (let i = 0; i < len; i++) {
                    binary += String.fromCharCode(bytes[i]);
                }
                return window.btoa(binary);
            }

            function base64ToArrayBuffer(base64) {
                const binary_string = window.atob(base64);
                const len = binary_string.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) {
                    bytes[i] = binary_string.charCodeAt(i);
                }
                return bytes.buffer;
            }

            async function handleAnalysis() {
                if (!idToken) {
                    showError("You are not signed in. Please refresh the page and sign in again.");
                    return;
                }
                const rawText = textInput.value.trim();
                if (rawText.length < 50) {
                    showError("Please enter at least 50 characters for an accurate analysis.");
                    return;
                }
                setLoadingState(true);
                hideError();
                resultsArea.classList.remove('visible');

                try {
                    // --- HYBRID ENCRYPTION LOGIC (Unchanged) ---
                    const aesKey = await window.crypto.subtle.generateKey({
                        name: "AES-GCM",
                        length: 256
                    }, true, ["encrypt", "decrypt"]);
                    const iv = window.crypto.getRandomValues(new Uint8Array(12));
                    const textEncoder = new TextEncoder();
                    const encryptedTextBuffer = await window.crypto.subtle.encrypt({
                        name: "AES-GCM",
                        iv: iv
                    }, aesKey, textEncoder.encode(rawText));
                    const exportedAesKeyBuffer = await window.crypto.subtle.exportKey("raw", aesKey);
                    const rsaEncrypt = new JSEncrypt();
                    rsaEncrypt.setPublicKey(publicKey);
                    const encryptedAesKey = rsaEncrypt.encrypt(arrayBufferToBase64(exportedAesKeyBuffer));
                    if (!encryptedAesKey) throw new Error("RSA encryption failed.");

                    // --- Prepare payload with authentication token ---
                    const payload = {
                        token: idToken, // Authenticate the user
                        encrypted_key: encryptedAesKey,
                        iv: arrayBufferToBase64(iv),
                        encrypted_text: arrayBufferToBase64(encryptedTextBuffer),
                    };

                    const response = await fetch('https://ai-detect.oscarz.dev/detect', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(payload)
                    });

                    const response_payload = await response.json();
                    if (!response.ok) {
                        if (response.status === 429 && response_payload.quota_info) {
                            updateQuotaDisplay(response_payload.quota_info);
                        }
                        throw new Error(response_payload.error || 'Server Error');
                    }

                    // --- DECRYPT THE RESPONSE (Unchanged) ---
                    const iv_response = base64ToArrayBuffer(response_payload.iv);
                    const encrypted_response = base64ToArrayBuffer(response_payload.encrypted_response);
                    const decrypted_buffer = await window.crypto.subtle.decrypt({
                        name: "AES-GCM",
                        iv: iv_response
                    }, aesKey, encrypted_response);
                    const textDecoder = new TextDecoder();
                    const decrypted_json_string = textDecoder.decode(decrypted_buffer);
                    const final_results = JSON.parse(decrypted_json_string);

                    displayResults(final_results);

                } catch (error) {
                    console.error('Analysis failed:', error);
                    showError(
                        `An error occurred: ${error.message}. Please check your daily word limit or try again later.`
                    );
                } finally {
                    setLoadingState(false);
                }
            }

            function displayResults(data) {
                const humanityScore = Math.round(100 - data.overall_percentage);
                document.getElementById('overall-score').textContent = `${humanityScore}%`;
                const progressBar = document.getElementById('progress-bar');
                const circumference = 2 * Math.PI * progressBar.r.baseVal.value;
                progressBar.style.strokeDashoffset = circumference - (humanityScore / 100) * circumference;
                let summary = "The analysis suggests this text is likely human-written.";
                if (humanityScore < 25) summary = "The analysis strongly suggests this text is AI-generated.";
                else if (humanityScore < 60) summary =
                    "This text shows a mix of human and AI-like characteristics.";
                document.getElementById('summary-text').textContent = summary;
                if (data.linguistics) {
                    updateStatCard('perplexity', data.linguistics.perplexity);
                    updateStatCard('variation', data.linguistics.sentence_length_variation);
                    updateStatCard('readability', data.linguistics.readability_grade);
                    updateStatCard('lexical', data.linguistics.lexical_richness);
                }
                const highlightedTextEl = document.getElementById('highlighted-text');
                highlightedTextEl.innerHTML = '';
                (data.chunks || []).forEach(chunk => {
                    const span = document.createElement('span');
                    span.textContent = chunk.text;
                    span.className = chunk.type.toUpperCase() === 'AI' ? 'ai-text' : 'human-text';
                    highlightedTextEl.appendChild(span);
                });

                if (data.quota_info) {
                    updateQuotaDisplay(data.quota_info);
                }

                resultsArea.classList.add('visible');
                resultsArea.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }

            function updateQuotaDisplay(quotaInfo) {
                if (quotaInfo.is_unlimited) {
                    userQuotaEl.textContent = 'Unlimited Access';
                } else {
                    userQuotaEl.textContent = `${quotaInfo.words_used_today} / ${quotaInfo.daily_limit} words`;
                }
            }

            // --- Other functions (updateStatCard, setLoadingState, etc.) are unchanged ---
            function updateStatCard(type, value) {
                const card = document.getElementById(`stat-card-${type}`);
                if (!card || value === null || value === undefined) return;
                const valueEl = card.querySelector('.stat-value');
                const ratingEl = card.querySelector('.stat-rating');
                const gaugeBarEl = card.querySelector('.gauge-bar');
                valueEl.textContent = value;
                let rating = {
                    text: 'Medium',
                    className: 'medium',
                    gaugePercent: 50
                };
                switch (type) {
                    case 'perplexity':
                        if (value > 150) rating = {
                            text: 'High',
                            className: 'high',
                            gaugePercent: 90
                        };
                        else if (value < 60) rating = {
                            text: 'Low',
                            className: 'low',
                            gaugePercent: 20
                        };
                        break;
                    case 'variation':
                        if (value > 6) rating = {
                            text: 'High',
                            className: 'high',
                            gaugePercent: 90
                        };
                        else if (value < 3) rating = {
                            text: 'Low',
                            className: 'low',
                            gaugePercent: 20
                        };
                        break;
                    case 'readability':
                        if (value < 8 || value > 13) rating = {
                            text: 'Natural',
                            className: 'high',
                            gaugePercent: 75
                        };
                        else rating = {
                            text: 'Common AI',
                            className: 'low',
                            gaugePercent: 30
                        };
                        break;
                    case 'lexical':
                        if (value > 0.65) rating = {
                            text: 'Rich',
                            className: 'high',
                            gaugePercent: 85
                        };
                        else if (value < 0.5) rating = {
                            text: 'Simple',
                            className: 'low',
                            gaugePercent: 35
                        };
                        else rating = {
                            text: 'Normal',
                            className: 'medium',
                            gaugePercent: 60
                        };
                        break;
                }
                ratingEl.textContent = rating.text;
                ratingEl.className = `stat-rating ${rating.className}`;
                gaugeBarEl.className = `gauge-bar ${rating.className}`;
                gaugeBarEl.style.width = `${rating.gaugePercent}%`;
            }

            function setLoadingState(isLoading) {
                checkButton.disabled = isLoading;
                const icon = checkButton.querySelector('i');
                document.getElementById('button-text').textContent = isLoading ? 'Analyzing...' : 'Analyze Text';
                icon.className = isLoading ? 'spinner' : 'fa-solid fa-wand-magic-sparkles';
            }

            function showError(message) {
                document.getElementById('error-message').textContent = message;
                document.getElementById('error-message').style.display = 'block';
            }

            function hideError() {
                document.getElementById('error-message').style.display = 'none';
            }

        });
