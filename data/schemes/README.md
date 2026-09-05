# Scheme Documents

Place publicly available Indian government scheme PDFs in this folder before running `python ingest.py`.

## Where to get official PDFs

Download only from official government sources, for example:

- **MyScheme** (national scheme discovery): https://www.myscheme.gov.in
- **India.gov.in — Schemes portal**: https://www.india.gov.in/topics/social-development/schemes
- **Ministry/department websites**: e.g., Ministry of Social Justice & Empowerment, Ministry of Education (scholarships via the National Scholarship Portal), MSME ministry for entrepreneurship schemes.
- **State government portals**: e.g., Karnataka's SSP scholarship portal or your own state's welfare department site.

Good candidates: central/state scholarship schemes, women-entrepreneur schemes, farmer welfare schemes.

## Requirements

- Text-based PDFs only. Scanned/image PDFs contain no extractable text and will be rejected by ingestion (OCR is out of scope for this MVP).
- File names become the "scheme name" shown in sources, so use descriptive names like `post_matric_scholarship.pdf`.
