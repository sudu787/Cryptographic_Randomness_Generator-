# References and Verification Log

Every entry below was checked against a primary or authoritative listing (publisher, standards body,
proceedings, or the authors' own page) on **2026-10-26**. The "Checked against" column states what was
seen. A DOI is given only where it was confirmed; entries without a DOI were not assigned one in the
source consulted, or the DOI was not confirmed and is deliberately omitted.

This list is the working bibliography for Phase 1. The full literature review (report section 7) will
extend it in Phase 6 using the same verification rule: **no citation enters the report unless its title,
authors, venue, and year have been checked.**

## Standards and specifications

| Key | Reference | Checked against |
|---|---|---|
| NIST-90A | E. Barker, J. Kelsey. *Recommendation for Random Number Generation Using Deterministic Random Bit Generators.* NIST SP 800-90A Rev. 1, June 2015. doi:10.6028/NIST.SP.800-90Ar1 | CSRC publication page. NIST issued a pre-draft call for comments on a Rev. 2 in September 2025; Rev. 1 remains the current version as of this check. |
| NIST-90B | M. S. Turan, E. Barker, J. Kelsey, K. McKay, M. Baish, M. Boyle. *Recommendation for the Entropy Sources Used for Random Bit Generation.* NIST SP 800-90B, January 2018. doi:10.6028/NIST.SP.800-90B | CSRC publication page and NIST release announcement (10 Jan 2018). CSRC notes (29 May 2025) that two errata have been identified for a future update. Section 4.4 (RCT, APT, Table 2) and the start-up test requirement (≥ 1024 samples) were read in the full text on 2026-10-26. |
| NIST-90C | E. Barker, J. Kelsey, K. McKay, A. Roginsky, M. S. Turan. *Recommendation for Random Bit Generator (RBG) Constructions.* NIST SP 800-90C, September 2025. doi:10.6028/NIST.SP.800-90C | CSRC publication page and NIST announcement (25 Sep 2025). Specifies RBG1, RBG2, RBG3 and RBGC constructions. |
| NIST-22 (full text read 2026-10-27, sections 2.1–2.4, 2.13, 3.4) | A. Rukhin, J. Soto, J. Nechvatal, M. Smid, E. Barker, S. Leigh, M. Levenson, M. Vangel, D. Banks, A. Heckert, J. Dray, S. Vo; revised by L. Bassham. *A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications.* NIST SP 800-22 Rev. 1a, April 2010. doi:10.6028/NIST.SP.800-22r1a | NIST publication record and the PDF title page (author order differs between the two; the PDF title page order is used here). NIST announced a decision to revise it on 19 Apr 2022; one stated goal is to reject its use for assessing cryptographic RNGs. No revised draft was found. |
| BSI-AIS2031 | W. Killmann, W. Schindler (BSI). *A Proposal for Functionality Classes for Random Number Generators* (AIS 20/31 reference), Version 3.0, 17 Sep 2024. | BSI download page. Version 3.0 allows min-entropy claims (class PTG.3). |
| RFC8439 | Y. Nir, A. Langley. *ChaCha20 and Poly1305 for IETF Protocols.* RFC 8439, June 2018. | RFC text (rfc-editor.org). Obsoletes RFC 7539. §2.3 sizes (256-bit key, 96-bit nonce, 32-bit counter; 2^32 blocks per (key, nonce)) and test vectors §2.3.2, §2.4.2 and A.1 #1–#3 were transcribed and cross-checked against the `cryptography` library on 2026-10-26; all match. |
| PYCA-CHACHA | pyca/cryptography documentation, `algorithms.ChaCha20`. https://cryptography.io/en/latest/hazmat/primitives/symmetric-encryption/ | Documentation page. The 16-byte nonce argument is the 4-byte little-endian block counter followed by the 12-byte RFC nonce. An initial counter n allows (2^32 − n)·64 bytes, after which `ValueError` is raised (observed with version 50.0.2). |
| POSIX-DRAND48 | IEEE Std 1003.1-2024 (POSIX Issue 8), `drand48` family. https://pubs.opengroup.org/onlinepubs/9799919799/functions/drand48.html | Specification page. a = 0x5DEECE66D, c = 0xB, m = 2^48; `mrand48()`/`jrand48()` return floor(X/2^16), the top 32 bits. |
| PEP506 | S. D'Aprano. *PEP 506 – Adding A Secrets Module To The Standard Library.* 2015, Final, Python 3.6. | peps.python.org. |
| CPY-URANDOM | CPython `os.urandom` implementation notes: on Windows, BCryptGenRandom replaced the deprecated CryptGenRandom in Python 3.11 (bpo-44611). On Linux, `getrandom()` is used (since 3.5; blocking mode until initialised since 3.6). | CPython commit for bpo-44611 and python-docs list archive. The live `os.urandom` docs page will be re-read before Phase 2's documentation claims are finalised. |

## Incidents and attacks motivating the project

| Key | Reference | Checked against |
|---|---|---|
| Debian-2008 | S. Yilek, E. Rescorla, H. Shacham, B. Enright, S. Savage. *When Private Keys are Public: Results from the 2008 Debian OpenSSL Vulnerability.* ACM Internet Measurement Conference (IMC), 2009. Background: CVE-2008-0166. | Authors' PDF (cs.utexas.edu). The IMC 2009 venue is from the authors' citation; the search results did not show the ACM page itself. |
| Heninger-2012 | N. Heninger, Z. Durumeric, E. Wustrow, J. A. Halderman. *Mining Your Ps and Qs: Detection of Widespread Weak Keys in Network Devices.* 21st USENIX Security Symposium, Bellevue, WA, Aug 2012, pp. 205–220. | USENIX proceedings page (Best Paper). |
| Ristenpart-2010 | T. Ristenpart, S. Yilek. *When Good Randomness Goes Bad: Virtual Machine Reset Vulnerabilities and Hedging Deployed Cryptography.* NDSS 2010. | NDSS listing and authors' page. |
| Everspaugh-2014 | A. Everspaugh, Y. Zhai, R. Jellinek, T. Ristenpart, M. Swift. *Not-So-Random Numbers in Virtualized Linux and the Whirlwind RNG.* IEEE Symposium on Security and Privacy, 2014, pp. 559–574. doi:10.1109/SP.2014.42 | CPS-VO bibliography and authors' page. |
| Markettos-2009 | A. T. Markettos, S. W. Moore. *The Frequency Injection Attack on Ring-Oscillator-Based True Random Number Generators.* CHES 2009, LNCS 5747, pp. 317–331. | Authors' PDF and Cambridge technical report publication list. |
| HurleySmith-2018 | D. Hurley-Smith, J. Hernandez-Castro. *Certifiably Biased: An In-Depth Analysis of a Common Criteria EAL4+ Certified TRNG.* IEEE Trans. Information Forensics and Security 13(4):1031–1041, 2018. | Royal Holloway research portal and IACR ePrint 2017/842. |

## Generator design and analysis

| Key | Reference | Checked against |
|---|---|---|
| Gutterman-2006 | Z. Gutterman, B. Pinkas, T. Reinman. *Analysis of the Linux Random Number Generator.* IEEE Symposium on Security and Privacy, 2006. | IACR ePrint 2006/086. |
| Dodis-2013 | Y. Dodis, D. Pointcheval, S. Ruhault, D. Vergnaud, D. Wichs. *Security Analysis of Pseudo-Random Number Generators with Input: /dev/random is not Robust.* ACM CCS 2013, pp. 647–658. doi:10.1145/2508859.2516653 | IACR ePrint 2013/338 BibTeX and DBLP. |
| Bernstein-2008 | D. J. Bernstein. *ChaCha, a variant of Salsa20.* Workshop Record of SASC 2008. | Author's page (cr.yp.to), paper dated 28 Jan 2008. |

## Entropy estimation, health testing, and statistical testing

| Key | Reference | Checked against |
|---|---|---|
| Kelsey-2015 | J. Kelsey, K. McKay, M. S. Turan. *Predictive Models for Min-Entropy Estimation.* CHES 2015, LNCS 9293, pp. 372–393. doi:10.1007/978-3-662-48324-4_19 | NIST publication record (BibTeX/RIS) and IACR ePrint 2015/600. |
| Kelsey-2024 | J. Kelsey. *Overview of SP 800-90.* Presentation, ECW 2024 (RNG/PUF session), 20 Nov 2024. https://www.creachlabs.fr/sites/default/files/public/media/document/2024-12/ecw2024-rng-puf-kelseyjohn-20241120.pdf | Slides, via search summary. States that the next SP 800-90B revision should derive health tests from the source's stochastic model and failure modes. The slide text will be read directly before it is quoted in the report. |
| Veljkovic-2012 | F. Veljković, V. Rožić, I. Verbauwhede. *Low-Cost Implementations of On-the-Fly Tests for Random Number Generators.* DATE 2012. | DATE 2012 proceedings PDF. |
| Yang-2015 | B. Yang, V. Rožić, N. Mentens, W. Dehaene, I. Verbauwhede. *Embedded HW/SW Platform for On-the-Fly Testing of True Random Number Generators.* DATE 2015. doi:10.7873/DATE.2015.0288 | DATE 2015 proceedings page. |
| Yang-2016 | B. Yang et al. *TOTAL: TRNG On-the-fly Testing for Attack detection using Lightweight hardware.* DATE 2016. | DATE 2016 proceedings PDF (search listing). The full author list will be copied from the PDF in Phase 6. |
| Rozic-2016 | V. Rožić, B. Yang, N. Mentens, I. Verbauwhede. *Canary Numbers: Design for Light-weight Online Testability of True Random Number Generators.* IACR ePrint 2016/386. | IACR ePrint. |
| Lin-2026 | C.-P. Lin. *A Unified NIST SP 800-90B Validation Framework for CMOS True Random Number Generators and Quantum Random Number Generators.* International Journal of Computer Applications 187(91), March 2026, starting p. 34. | Full text (IJCA PDF). Closest related work found so far; see the novelty assessment in [proposal.md](proposal.md). |
| DavidBarton-1962 | F. N. David, D. E. Barton. *Combinatorial Chance.* Hafner, New York, 1962, p. 230. | Cited in SP 800-22 §3.4 as the source of the longest-run formula, whose text was read in SP 800-22. The book itself was not consulted; the formula is checked numerically (`tests/test_statistics.py`). |
| Riesinger-2025 | Riesinger, Fuß. *Testing the Tests: Opportunities for Corrections and Improvements …* (technical audit of NIST SP 800-22r1a). IACR ePrint 2025/856. | IACR ePrint listing and abstract. The PDF returned HTTP 403 on 2026-10-27, so the paper's specific findings were not compared with this project's (methodology §9.1). Reports imprecise constants and erroneous worked examples in SP 800-22r1a. **Consequence for this project:** SP 800-22 worked examples cannot be used as test vectors unless they are independently recomputed. The full title and authors' given names will be copied from the paper in Phase 6. |
| LEcuyer-2007 | P. L'Ecuyer, R. Simard. *TestU01: A C Library for Empirical Testing of Random Number Generators.* ACM Trans. Math. Softw. 33(4), Article 22, 2007. doi:10.1145/1268776.1268777 | Paper header (DOI) and secondary listings for volume and issue. Publisher page not opened. |

## Pseudorandom generator theory (to be checked against publisher pages in Phase 6)

- T. E. Hull, A. R. Dobell (1962), *Random Number Generators.* SIAM Review: the full-period
  conditions implemented in `has_full_period`.
- W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery (1992), *Numerical Recipes in C*,
  2nd ed., Cambridge University Press, ch. 7: source of the 32-bit LCG parameters a = 1664525,
  c = 1013904223. The parameters are widely reproduced; the exact section has not yet been checked.
- D. E. Knuth (1997), *The Art of Computer Programming*, Vol. 2, 3rd ed., §3.2: LCG theory.

## Statistical methodology (standard references, used for interval estimation and multiple testing)

These are classical results. Their bibliographic details will be checked against publisher pages in
Phase 6 before they are cited in the report.

- E. B. Wilson (1927), score interval for a binomial proportion. *Journal of the American Statistical Association.*
- C. J. Clopper, E. S. Pearson (1934), exact binomial confidence limits. *Biometrika.*
- S. Holm (1979), sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics.*
- Y. Benjamini, Y. Hochberg (1995), false discovery rate control. *Journal of the Royal Statistical Society, Series B.*
- B. Efron, R. Tibshirani (1993), *An Introduction to the Bootstrap.* Chapman & Hall.
