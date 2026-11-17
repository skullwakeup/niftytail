# NIFTYTAIL — Data record (Day 3B/3C)

**Updated:** 2026-11-12. This file contains no index levels, OHLC values or per-day returns (D23, D26). It contains only file names, dates, counts, categories and hashes. `tests/test_data.py::test_data_doc_has_no_index_levels` enforces this.

**Labels used below.** EXPLAINED means a public source states the reason, and the URL is cited. UNEXPLAINED means no public source was found in this work package. UNVERIFIED means not checked against any source. The labels describe evidence only and decide nothing.

**Source-verification level.** Every URL below was identified through web-search results on 2026-11-11. The cited pages were **not opened** in this work package: fetching nseindia.com and niftyindices.com pages timed out. Each label therefore rests on the search engine's summary of the cited page. The owner may want to open the pages before relying on a label.

## 1. Source and series definition

| Item | Value |
|---|---|
| Series | Nifty 50 **Price Return** index (D2), field **Close** |
| Source page | https://www.niftyindices.com/reports/historical-data |
| Retrieved | 2026-11-11, by the owner |
| Acquisition | manual download for local use (D23); 32 unedited CSV files in `data/raw/` |
| Manifest | `data/raw/MANIFEST.toml`, written by `validate --init-manifest` (schema 1; series PR; field Close) |
| Sample start | 1995-11-03, the official base date (D1) |

**Terms of use.** Wording recorded in plan §5b, read on 2026-11-11:

- *Terms of Use* (niftyindices.com/terms-of-use): "The Site may be used only for lawful purposes by individuals for personal, non-commercial use only." and "No material from the Site may be copied, modified, reproduced, republished, uploaded, transmitted, posted or distributed in any form without prior written permission from the Company."
- *Disclaimer* (niftyindices.com/disclaimer): "No part of this Site or information contained herein may be reproduced, stored in a retrieval system or transmitted in any form or by any means … without prior written permission of the Company or its applicable group company." and "Use or distribution of the Company's … index data and the use of their index data to create financial products require a license from NSE Indices Limited and/or its third party licensors."

**Re-verification status: not re-verified.** On 2026-11-11, fetching https://www.niftyindices.com/terms-of-use, https://niftyindices.com/terms-of-use and https://www.niftyindices.com/disclaimer timed out (60 s each).

## 2. File format (Day-3A structure inspection)

- One CSV per one-year window, named `NIFTY 50_Historical_PR_DDMMYYYYtoDDMMYYYY.csv`. The windows are contiguous and non-overlapping, and the last one is partial.
- UTF-8 without BOM, CRLF line endings, no trailing newline, every field double-quoted, comma delimiter.
- The header is exactly `"Index Name","Date","Open","High","Low","Close"`, and `Index Name` is always `NIFTY 50`.
- `Date` is `dd Mon yyyy` (English month abbreviation). Rows run newest first.
- Numbers have no thousands separators. `Close` is always a plain decimal. `Open`/`High`/`Low` hold a `-` placeholder in one row of the first file (3 cells).

## 3. Inventory

The total is 32 files, 7,717 rows and 516,207 bytes. The SHA-256 values are recomputed on 2026-11-11 and match `MANIFEST.toml` (32/32).

| name | rows | first_date | last_date | bytes | sha256 |
|---|---|---|---|---|---|
| NIFTY 50_Historical_PR_03111995to02111996.csv | 251 | 1995-11-03 | 1996-11-01 | 16099 | 42c7a4cdf217ba12cbc03edf7608d42c4a3dd7465bc87d4933446bf43c713e80 |
| NIFTY 50_Historical_PR_03111996to02111997.csv | 242 | 1996-11-04 | 1997-10-30 | 15726 | d3e4ecc51a281d71a8199e8ed1f947a44b5fef74acdedbab2d11d331b87bc893 |
| NIFTY 50_Historical_PR_03111997to02111998.csv | 249 | 1997-11-03 | 1998-11-02 | 15951 | 141a009e28fa1d9f3b1bb1f70b5e7875dc4f0b94aa6a7b2194965e6cac4835f7 |
| NIFTY 50_Historical_PR_03111998to02111999.csv | 255 | 1998-11-03 | 1999-11-02 | 16492 | da44de03c94ca972bd4368fa3f07e73ad5bbe963fe2337c5d7a9b0d3d61d6bdd |
| NIFTY 50_Historical_PR_03111999to02112000.csv | 251 | 1999-11-03 | 2000-11-02 | 16613 | 276e139ff187256de199594d15c763a13e9c2a1c76b655365a906c174e4ba011 |
| NIFTY 50_Historical_PR_03112000to02112001.csv | 251 | 2000-11-03 | 2001-11-02 | 16473 | 5fd5e27adbd7cf0ffbe6445cd8ac06622f7c202fa6f73b70e7ef014cec0e9151 |
| NIFTY 50_Historical_PR_03112001to02112002.csv | 249 | 2001-11-05 | 2002-11-01 | 16227 | 5d8095c0b24c75c7c16de50e143d52a067f0e353658c4ce864cdf7af548f47fd |
| NIFTY 50_Historical_PR_03112002to02112003.csv | 251 | 2002-11-04 | 2003-10-31 | 16421 | 60bbcb5a436b2450b38ddb4cba4124826ba852c365a21b9e3a363e6a3d5ea7ee |
| NIFTY 50_Historical_PR_03112003to02112004.csv | 255 | 2003-11-03 | 2004-11-02 | 16877 | 8794dade3ef6139b5b5d7f99c31be11edf3bc0ebfb6a4170d0b1e42f70b16fa8 |
| NIFTY 50_Historical_PR_03112004to02112005.csv | 252 | 2004-11-03 | 2005-11-02 | 16679 | 2ed6f61aced7840341d39612c5c465a4ea8683152bfa96a7a1e4412a48c1ed45 |
| NIFTY 50_Historical_PR_03112005to02112006.csv | 250 | 2005-11-07 | 2006-11-02 | 16547 | d572e74edff4995dd2d929cb57ae5454b35a4d038e49d7a5d6eb3faa8dca6f17 |
| NIFTY 50_Historical_PR_03112006to02112007.csv | 250 | 2006-11-03 | 2007-11-02 | 16547 | c2dfc95ad93c61480e56c9f4363908ca90a18de646377712656de23fe2e78907 |
| NIFTY 50_Historical_PR_03112007to02112008.csv | 246 | 2007-11-05 | 2008-10-31 | 16283 | e598afd3658ef2b859a650a48dddc28f382ae13af17f58668e415398dc66ecc7 |
| NIFTY 50_Historical_PR_03112008to02112009.csv | 241 | 2008-11-03 | 2009-10-30 | 15953 | 4f092bf03c215f8056960661cb636c1aa7a4d3a0b965169a7ab1651e9dd73619 |
| NIFTY 50_Historical_PR_03112009to02112010.csv | 252 | 2009-11-03 | 2010-11-02 | 16679 | b3958320c35630902b3bc37d1aea9e09afc3735b6f584b30658977d6c1055b4e |
| NIFTY 50_Historical_PR_03112010to02112011.csv | 249 | 2010-11-03 | 2011-11-02 | 16481 | f127f63c0079c0717a0a195d08653172e435ef13af282d009e22f23239fb6a47 |
| NIFTY 50_Historical_PR_03112011to02112012.csv | 252 | 2011-11-03 | 2012-11-02 | 16679 | 1bd60a198cb251dcb87c501bb76519f732905da94ed90f2e70f240e31b0a9189 |
| NIFTY 50_Historical_PR_03112012to02112013.csv | 248 | 2012-11-05 | 2013-11-01 | 16415 | dc293153009e01eb107ef0cdaaecb5a3c9b1422856e261e79e4d1cbae1abb6db |
| NIFTY 50_Historical_PR_03112013to02112014.csv | 244 | 2013-11-03 | 2014-10-31 | 16151 | 426f7eb218ef826a1dc0bd7b60097e37a64527169e0ed6fb322b727ad53b9e1f |
| NIFTY 50_Historical_PR_03112014to02112015.csv | 248 | 2014-11-03 | 2015-11-02 | 16415 | 7b6bd9a7167216fdd0e5940d9f781251f522bcf1b08f8b2f9ec9dcde7c6b34e3 |
| NIFTY 50_Historical_PR_03112015to02112016.csv | 246 | 2015-11-03 | 2016-11-02 | 16283 | 9d4eee5b9b1a7a6a4902776b19768f0e21e3432b83ccb32f70e1a6cd5c6d4381 |
| NIFTY 50_Historical_PR_03112016to02112017.csv | 249 | 2016-11-03 | 2017-11-02 | 16617 | 4e4a46f2641b45a18d442106caec065eb320d0a6ae7bc070bd595ed89ef383f5 |
| NIFTY 50_Historical_PR_03112017to02112018.csv | 248 | 2017-11-03 | 2018-11-02 | 17402 | 657c58f8a952668cf3f1bc27a9d89ee8cfd80ac281c17cbadc479cb6932fe811 |
| NIFTY 50_Historical_PR_03112018to02112019.csv | 243 | 2018-11-05 | 2019-11-01 | 17057 | 520b6c8bcada90b615d34bc0779de7c802250de7bcdd08f90cefae1e24bc184d |
| NIFTY 50_Historical_PR_03112019to02112020.csv | 251 | 2019-11-04 | 2020-11-02 | 17390 | a4c93ca86cbc1fa01c1babd51e34ebf44605cc61dcbfd7c0f6b8ddc9e8ef4b36 |
| NIFTY 50_Historical_PR_03112020to02112021.csv | 248 | 2020-11-03 | 2021-11-02 | 17407 | 2ec5d0773adf49292bed7a7e1840aeea719fbdda8cf966d1adc98ee566759b01 |
| NIFTY 50_Historical_PR_03112021to02112022.csv | 248 | 2021-11-03 | 2022-11-02 | 17407 | ba1a328018db81750dded330f8f8f6519428e6b5dc4a0ea1f8e2c843751c6922 |
| NIFTY 50_Historical_PR_03112022to02112023.csv | 248 | 2022-11-03 | 2023-11-02 | 17407 | 0d0dffdedcc75fee4cd5881e87677b822b285563a967d94c1e7a5d62d1eb96f1 |
| NIFTY 50_Historical_PR_03112023to02112024.csv | 249 | 2023-11-03 | 2024-11-01 | 17228 | 945dbb8e4f3c753e39fd3b58974b40fd84169df01d0bd22ee422b4ffadac3382 |
| NIFTY 50_Historical_PR_03112024to02112025.csv | 247 | 2024-11-04 | 2025-10-31 | 16900 | 4407e66ab343d00bd475c6f320150100fe348de7028029c6ad9eea2d14284253 |
| NIFTY 50_Historical_PR_03112025to02112026.csv | 247 | 2025-11-03 | 2026-11-02 | 16872 | 771a26ea5268fa31224164725b22b6ff3ee8c09a1ef273f822b839ec1b6badc0 |
| NIFTY 50_Historical_PR_03112026to11112026.csv | 7 | 2026-11-03 | 2026-11-11 | 529 | a34a92d4e1b4355e0d88f5e18d8fc39be41918d711e4df697fd0243ed2b6373c |

## 4. Validation result

Command: `uv run --locked python -m niftytail validate configs\base.toml --init-manifest --retrieved 2026-11-11 --source-url https://www.niftyindices.com/reports/historical-data`. It exited with status 0 ("validation passed"). A rerun without `--init-manifest` produced byte-identical outputs and an unchanged manifest. A second `--init-manifest` was refused.

| Quantity | Value | Day-3A inspection |
|---|---|---|
| hard errors | 0 | — |
| n_files | 32 | 32 |
| n_rows | 7,717 | 7,717 |
| first_date / last_date | 1995-11-03 / 2026-11-11 | same |
| weekend_date flags | 42 | 42 |
| missing_open / missing_high / missing_low | 1 / 1 / 1 (one row, 3 cells) | 3 cells |
| calendar_gap flags (≥ 5 days) | 15 | — |
| large_return flags (\|log return\| ≥ 0.05) | 80 | — |
| stale_close_run flags (≥ 3) | 0 | — |
| identical_close_pair flags (exactly 2; D28, added Day 3C) | 8 | — |
| ohlc_inconsistent flags | 0 | — |
| absent weekdays (D26) | 419 | — |
| first_close_equals_base_value | true | — |

All outputs are in `data/validation/` (gitignored): `validation_report.json`, `flags.csv`, `spot_check_sheet.txt` and `absent_weekdays.csv`.

## 5. Audits

### 5.1 Weekend dates (42)

Of the 42 dates, 29 are EXPLAINED and 13 are UNEXPLAINED. Two of them are also large_return flags. None of the 8 identical_close_pair flags (D28) involves a weekend-dated row, so no weekend row repeats the previous close. No pair involves any of the 13 UNEXPLAINED weekend dates.

| date | day | label | reason (per source) | source |
|---|---|---|---|---|
| 1996-11-10 | Sun | UNEXPLAINED | — | none found |
| 1997-03-01 | Sat | UNEXPLAINED | — | none found |
| 1997-04-12 | Sat | UNEXPLAINED | — | none found |
| 1998-10-31 | Sat | UNEXPLAINED | — | none found |
| 1998-11-21 | Sat | UNEXPLAINED | — | none found |
| 1998-11-28 | Sat | UNEXPLAINED | — | none found |
| 1999-02-27 | Sat | EXPLAINED | Union Budget presented on a Saturday; markets open | https://www.business-standard.com/article/markets/markets-will-stay-open-on-budget-day-hints-sebi-115022001010_1.html ; https://www.freepressjournal.in/india/budget-2026-sunday-trading-returns-after-27-years-as-markets-open-for-only-2nd-time |
| 1999-03-20 | Sat | UNEXPLAINED | — | none found |
| 1999-04-17 | Sat | EXPLAINED | special trading session on BSE and NSE (RBI footnote) | https://website.rbi.org.in/hi/web/rbi/-/publications/weekly-statistical-supplement/15.-bse-sensitive-index-and-nse-nifty-index-of-ordinary-share-prices-mumbai-773 |
| 1999-10-23 | Sat | UNEXPLAINED | — | none found |
| 1999-11-07 | Sun | UNEXPLAINED | — | none found |
| 2003-03-22 | Sat | EXPLAINED | NSE live session from its back-up site | https://nsearchives.nseindia.com/content/press/20032003.pdf |
| 2003-10-25 | Sat | EXPLAINED | Diwali Muhurat session | https://nsearchives.nseindia.com/content/press/20102003.pdf |
| 2003-11-15 | Sat | UNEXPLAINED | a special Saturday session is mentioned in one article, but its date is not confirmed | https://www.valueresearchonline.com/stories/6998/stock-markets-falter/ (undated) |
| 2004-04-17 | Sat | UNEXPLAINED | — | none found |
| 2004-10-09 | Sat | UNEXPLAINED | — | none found |
| 2005-06-04 | Sat | EXPLAINED | 3-hour special session to test NSE back-up facilities at Chennai (RBI WSS footnote) | https://website.rbi.org.in/web/rbi/-/publications/weekly-statistical-supplement/15.-bse-sensitive-index-and-nse-nifty-index-of-ordinary-share-prices-mumbai-8841 |
| 2005-11-26 | Sat | EXPLAINED | 3-hour special session from the Disaster Recovery Site (RBI WSS footnote) | https://website.rbi.org.in/web/rbi/-/publications/weekly-statistical-supplement/weekly-traded-volume-in-corporate-debt-at-nse-9466 |
| 2006-04-29 | Sat | UNEXPLAINED | — | none found |
| 2006-06-25 | Sun | EXPLAINED | special live session (software update) | https://business-standard.com/article/companies/stock-markets-open-on-sunday-106062300017_1.html ; https://niftyindices.com/Press_Release/ind_prs21062006.pdf |
| 2006-10-21 | Sat | EXPLAINED | Diwali Muhurat session | https://business-standard.com/article/companies/bse-nse-muhurat-trading-on-oct-21-106101000045_1.html |
| 2009-10-17 | Sat | EXPLAINED | Diwali Muhurat session | https://www.business-standard.com/article/markets/muhurat-trading-from-1815-hrs-on-bse-nse-109101700101_1.html |
| 2010-02-06 | Sat | EXPLAINED | 90-minute special session (trading-system upgrade) | https://business-standard.com/article/markets/bse-nse-to-hold-special-trading-on-february-6-110020500049_1.html ; https://taxguru.in/sebi/nse-bse-to-hold-90-minute-special-trading-session-on-saturday.html |
| 2012-01-07 | Sat | EXPLAINED | special live session (F&O system upgrade) | https://www.business-standard.com/article/markets/special-f-o-trading-sessions-on-saturday-112010500188_1.html ; https://www.cse-india.com/upload/CSE%20Notices%20&%20Circulars/2012/Notice050112_1.htm |
| 2012-03-03 | Sat | EXPLAINED | special session (disaster-recovery test) | https://www.businesstoday.in/story/stock-markets-special-session-mar-3-2012/1/22892.html ; https://www.cse-india.com/upload/CSE%20Notices%20&%20Circulars/2012/Notice050312.htm |
| 2012-04-28 | Sat | EXPLAINED | special live session (NSE F&O hardware upgrade) | https://www.cse-india.com/upload/CSE%20Notices%20&%20Circulars/2012/Notice260412.htm |
| 2012-09-08 | Sat | EXPLAINED | special session (BSE disaster-recovery test; NSE also open) | https://www.business-standard.com/article/markets/bse-nse-to-hold-special-trading-on-sep-8-112082700143_1.html |
| 2013-05-11 | Sat | EXPLAINED | special session (disaster-recovery site) | https://www.business-standard.com/article/markets/markets-end-flat-in-special-trading-session-113051100178_1.html |
| 2013-11-03 | Sun | EXPLAINED | Diwali Muhurat session | https://www.business-standard.com/article/news-cm/muhurat-trading-on-3-november-2013-113101900380_1.html |
| 2014-03-22 | Sat | EXPLAINED | special live session | https://www.business-standard.com/article/markets/markets-to-open-today-for-a-two-hour-special-trading-session-114032200129_1.html ; https://www.cse-india.com/upload/CSE%20Notices%20&%20Circulars/2014/Notice200314_4.htm |
| 2015-02-28 | Sat | EXPLAINED | Union Budget day session | https://www.business-standard.com/article/reuters/indian-share-markets-to-open-on-budget-day-on-feb-28-115022001365_1.html |
| 2016-10-30 | Sun | EXPLAINED | Diwali Muhurat session (secondary source) | https://www.finnovate.in/learn/blog/20-years-muhurat-trading-historical-returns |
| 2019-10-27 | Sun | EXPLAINED | Diwali Muhurat session | https://nsearchives.nseindia.com/content/circulars/CMTR42483.pdf ; https://zerodha.com/marketintel/bulletin/234057/muhurat-trading-session-on-account-of-diwali-3 |
| 2020-02-01 | Sat | EXPLAINED | Union Budget day session | https://www.business-standard.com/article/markets/shares-bought-today-cannot-be-sold-on-the-budget-day-here-s-why-120013100335_1.html |
| 2020-11-14 | Sat | EXPLAINED | Diwali Muhurat session (secondary sources) | https://www.businesstoday.in/markets/market-perspective/bse-nse-to-hold-muhurat-trading-on-nov-14/story/421092.html ; https://anandrathi.com/blog/muhurat-trading-last-10-years-nifty-sensex-performance |
| 2023-11-12 | Sun | EXPLAINED | Diwali Muhurat session (secondary source) | https://www.editorji.com/business-news/personal-finance/diwali-muhurat-trading-2023-history-key-date-stock-market-timings-and-essential-details-to-note-1699620520557 |
| 2024-01-20 | Sat | EXPLAINED | special live session with switch to the DR site | https://www.angelone.in/news/share-market/nse-special-live-session-on-january-20-2024 |
| 2024-03-02 | Sat | EXPLAINED | special live session with switch to the DR site | https://www.business-standard.com/markets/stock-market-news/nse-to-hold-special-trading-session-on-mar-2-to-check-market-infrastructure-124021401376_1.html |
| 2024-05-18 | Sat | EXPLAINED | special live session with switch to the DR site | https://www.business-standard.com/markets/news/nse-to-run-special-trading-session-on-may-18-with-disaster-recovery-switch-124050701155_1.html |
| 2025-02-01 | Sat | EXPLAINED | Union Budget day session | https://www.kotaksecurities.com/bulletins/special-live-trading-session-on-saturday-february-1-2025/ |
| 2026-02-01 | Sun | EXPLAINED | Union Budget day session | https://nsearchives.nseindia.com/content/circulars/CMTR72349.pdf ; https://zerodha.com/marketintel/bulletin/439957/special-live-trading-session-on-sunday-february-1-2026 |
| 2026-11-08 | Sun | EXPLAINED | Diwali Muhurat session (NSE circular dated 2026-10-19) | https://nsearchives.nseindia.com/content/circulars/CMTR76941.pdf ; https://www.outlookmoney.com/invest/diwali-muhurat-trading-2026-nse-schedules-rare-afternoon-session-on-november-8-check-full-schedule |

### 5.2 Absent weekdays (D26)

An absent weekday is a Monday–Friday date strictly between 1995-11-03 and 2026-11-11 that has no row. Weekend rows are never absent. The audit counts 419 absent weekdays. Grouped into maximal runs (Fri → Mon adjacent), there are 375 single days, 18 runs of 2, 1 run of 3 and 1 run of 5.

| year | rows | weekend_rows | absent_weekdays |
|---|---|---|---|
| 1995 | 40 | 0 | 1 |
| 1996 | 250 | 1 | 13 |
| 1997 | 244 | 2 | 19 |
| 1998 | 250 | 3 | 14 |
| 1999 | 254 | 5 | 12 |
| 2000 | 250 | 0 | 10 |
| 2001 | 248 | 0 | 13 |
| 2002 | 251 | 0 | 10 |
| 2003 | 254 | 3 | 10 |
| 2004 | 254 | 2 | 10 |
| 2005 | 251 | 2 | 11 |
| 2006 | 250 | 3 | 13 |
| 2007 | 249 | 0 | 12 |
| 2008 | 246 | 0 | 16 |
| 2009 | 243 | 1 | 19 |
| 2010 | 252 | 1 | 10 |
| 2011 | 247 | 0 | 13 |
| 2012 | 251 | 4 | 14 |
| 2013 | 250 | 2 | 13 |
| 2014 | 244 | 1 | 18 |
| 2015 | 248 | 1 | 14 |
| 2016 | 247 | 1 | 15 |
| 2017 | 248 | 0 | 12 |
| 2018 | 246 | 0 | 15 |
| 2019 | 245 | 1 | 17 |
| 2020 | 252 | 2 | 12 |
| 2021 | 248 | 0 | 13 |
| 2022 | 248 | 0 | 12 |
| 2023 | 246 | 1 | 15 |
| 2024 | 249 | 3 | 16 |
| 2025 | 249 | 1 | 13 |
| 2026 | 213 | 2 | 14 |
| **total** | **7,717** | **42** | **419** |

1995 covers only 1995-11-03 to 1995-12-31, and 2026 covers only 2026-01-01 to 2026-11-11.

**Single-day absences (375): UNVERIFIED.** By decade: 1990s 50, 2000s 108, 2010s 124, 2020s 93. No NSE holiday calendar was available for most years, so these days are not "explained". They are listed in `data/validation/absent_weekdays.csv`.

**Runs of length ≥ 2 (20 runs).** Of these, 7 are EXPLAINED and 13 are UNEXPLAINED.

| run | length | label | reason (per source) | source |
|---|---|---|---|---|
| 1996-12-27 Fri, 1996-12-30 Mon | 2 | UNEXPLAINED | — | none found |
| 1997-10-02 Thu … 1997-10-08 Wed | 5 | EXPLAINED | NSE trading stopped after the INSAT-2D satellite failure; resumed on 1997-10-09 | https://www.business-standard.com/amp/article/specials/insat-2d-failure-cripples-nse-no-trading-today-197100301027_1.html ; https://www.business-standard.com/article/specials/no-nse-trading-till-thursday-no-major-impact-expected-197100601092_1.html |
| 1998-10-01 Thu, 1998-10-02 Fri | 2 | UNEXPLAINED | — | none found |
| 2000-03-17 Fri, 2000-03-20 Mon | 2 | UNEXPLAINED | — | none found |
| 2005-11-03 Thu, 2005-11-04 Fri | 2 | UNEXPLAINED | — | none found |
| 2006-10-24 Tue, 2006-10-25 Wed | 2 | UNEXPLAINED | — | none found |
| 2007-05-01 Tue, 2007-05-02 Wed | 2 | UNEXPLAINED | — | none found |
| 2008-03-20 Thu, 2008-03-21 Fri | 2 | UNEXPLAINED | — | none found |
| 2009-03-10 Tue, 2009-03-11 Wed | 2 | UNEXPLAINED | — | none found |
| 2009-04-30 Thu, 2009-05-01 Fri | 2 | UNEXPLAINED | — | none found |
| 2009-12-25 Fri, 2009-12-28 Mon | 2 | UNEXPLAINED | — | none found |
| 2011-08-31 Wed, 2011-09-01 Thu | 2 | UNEXPLAINED | — | none found |
| 2012-04-05 Thu, 2012-04-06 Fri | 2 | UNEXPLAINED | both dates appear as gazetted public holidays; no exchange source found | https://www.india-briefing.com/news/indias-official-2012-holiday-schedule-5603.html/ (public-holiday list, not an exchange source) |
| 2014-10-02 Thu, 2014-10-03 Fri, 2014-10-06 Mon | 3 | EXPLAINED | Gandhi Jayanti, Dussehra and Bakri Id market holidays | https://www.business-standard.com/article/markets/markets-closed-on-account-of-dussehra-114100300132_1.html ; https://businesstoday.in/story/stock-market-sensex-nifty-closed-oct-2-6-gandhi-jayanti-dussehra/1/210993.html |
| 2015-04-02 Thu, 2015-04-03 Fri | 2 | UNEXPLAINED | source found for 2015-04-02 only (Mahavir Jayanti); none for 2015-04-03 | https://www.business-standard.com/article/pti-stories/markets-holiday-note-115040200156_1.html |
| 2016-03-24 Thu, 2016-03-25 Fri | 2 | EXPLAINED | Holi; Good Friday | https://zerodha.com/z-connect/general/trading-holidays-2016-nse-bse-mcx |
| 2016-04-14 Thu, 2016-04-15 Fri | 2 | EXPLAINED | Ambedkar Jayanti; Ram Navami | https://zerodha.com/z-connect/general/trading-holidays-2016-nse-bse-mcx |
| 2016-10-11 Tue, 2016-10-12 Wed | 2 | EXPLAINED | Dussehra; Muharram | https://zerodha.com/z-connect/general/trading-holidays-2016-nse-bse-mcx |
| 2018-03-29 Thu, 2018-03-30 Fri | 2 | EXPLAINED | Mahavir Jayanti; Good Friday | https://www.business-standard.com/article/markets/markets-closed-on-thursday-friday-due-to-mahavir-jayanti-good-friday-118032800728_1.html |
| 2022-04-14 Thu, 2022-04-15 Fri | 2 | EXPLAINED | Ambedkar Jayanti / Mahavir Jayanti; Good Friday | https://www.indiainfoline.com/news/markets/stock-market-closed-on-april-14-and-15-on-account-of-mahavir-jayanti-dr-baba-saheb-ambedkar-and-good-friday |

Note: when the Zerodha page was fetched directly on 2026-11-11, it showed the 2026 list. Its 2016 content is known only from the search result.

### 5.3 Large returns (|log return| ≥ 0.05): 80 dates

By decade: 1990s 16, 2000s 52, 2010s 2, 2020s 10. The dates are listed only in `data/validation/flags.csv` (gitignored; D65 (j)).

**Market-wide halts and suspensions found in public sources.** The matches below are by date only, and no conclusion is drawn. Of the six single-day events, all six dates are in the series and four carry a large_return flag; the 1997 suspension is the absent run of §5.2. This list may be incomplete: one source says the market-wide circuit breaker was triggered only three times since 2001, which conflicts with the dates below.

| date | event (per source) | source |
|---|---|---|
| 1997-10-02 … 1997-10-08 | NSE trading suspended (INSAT-2D failure) | see §5.2 |
| 2004-05-17 | trading halted twice (index circuit filter) | https://eparlib.nic.in/bitstream/123456789/509912/1/1216.pdf ; https://www.domain-b.com/investments/markets/bse/20040518_sensex.html |
| 2008-01-22 | trading halted after the Nifty breached the 10% level | https://www.outlookbusiness.com/amp/story/equity/in-a-historic-volatile-session-circuit-breakers-applied-in-12-years-4493 |
| 2009-05-18 | upper circuit; trading halted for the rest of the day | https://business-standard.com/article/markets/sensex-frozen-trading-halts-for-the-day-109051800123_1.html |
| 2020-03-13 | lower circuit; 45-minute halt | https://www.businesstoday.in/markets/stocks/sensex-nifty-crash-circuit-breaker-rules/story/398162.html |
| 2020-03-23 | lower circuit; 45-minute halt | https://www.business-standard.com/article/markets/sensex-hits-10-lower-circuit-trading-halted-for-45-minutes-120032300237_1.html |
| 2021-02-24 | NSE halt for about 4 hours (technical glitch, not a circuit breaker) | https://www.business-standard.com/amp/article/news-cm/nse-halts-trading-due-to-technical-issues-121022400465_1.html |

### 5.4 Stale runs, calendar gaps, OHLC

- **stale_close_run (≥ 3 identical consecutive closes):** 0.
- **identical_close_pair (exactly 2 identical consecutive closes, dated at the second row; D28):** 8. All are on weekdays. By decade of the second row: 1990s 1, 2000s 3, 2010s 3, 2020s 1. The dates are listed only in `data/validation/flags.csv` (gitignored; D65 (j)).
- **ohlc_inconsistent:** 0. The one row with missing Open/High/Low is excluded from this check by construction.
- **calendar_gap (≥ 5 calendar days between consecutive rows):** 15. Each is the first row after the gap:
  1996-12-31 (5 days), 1997-10-09 (8), 1998-10-05 (5), 2000-03-21 (5), 2005-11-07 (5), 2008-03-24 (5), 2009-05-04 (5), 2009-12-29 (5), 2012-04-09 (5), 2014-10-07 (6), 2015-04-06 (5), 2016-03-28 (5), 2016-04-18 (5), 2018-04-02 (5), 2022-04-18 (5).
  Every gap corresponds to one of the absent-weekday runs in §5.2. The full list is in `data/validation/flags.csv`.

## 6. Sample size

| Quantity | Value |
|---|---|
| n_rows | 7,717 |
| n_returns = n_rows − 1 | 7,716 |
| T0 (`configs/base.toml` `[design] train_size`) | 2,000 |
| out-of-sample size n_returns − T0 | **5,716** |

The plan and the methodology use the approximation "≈ 5,650" in these places. Neither document was edited.

- `docs/plan_v2.2.md`: lines 31, 229, 233, 260, 266, 351 and 506.
- `docs/methodology.md`: lines 85, 93, 346, 353, 354, 370, 436, 444, 531, 542, 715, 716, 729–731, 743, 747, 767, 790 and 792.
- `configs/base.toml`: no hit. Line 29 holds `train_size = 2000`.
- No "7,650" or "7650" appears in any of the three files.

Arithmetic: 5,716 = 22 x 250 + 216. The 22 full traffic-light blocks are unchanged, but the partial block has 216 observations, not the 150 stated in `methodology.md:444`.

See D27 (5) for how the data-dependent numbers are reported at the actual T.

## 7. Base value and NSE methodology PDF

The owner supplied `data/docs/nifty50_methodology.pdf` (3,697,637 bytes, 345 pages; not tracked): *NSE Indices Limited, Methodology Document for Equity Indices, September 2026*. Text was extracted read-only with `pdftotext`. "p." is the PDF page; the printed page number is one lower.

| topic | page | quote (< 15 words) |
|---|---|---|
| base date and base value | p. 284 (Index Characteristics table) | "Nifty 50 Nov 03, 1995 1000 Free float" |
| computation | p. 20 | "computed using a float-adjusted, market capitalization weighted methodology" |
| computation history | p. 20 | "Beginning June 26, 2009, the Nifty 50 is being computed using float-adjusted" |
| index value formula | p. 332 | "Index Value = Index Market Capitalisation / Base Free Float Market Capitalisation of index" |
| closing value | p. 326 | "after the close of trading and after the calculation of the closing value" |

- **How the closing value is computed:** a text search found no statement of how the index's daily closing value is determined (for example, which constituent prices are used).
- **Circuit breakers / halts:** the phrases "circuit breaker", "market-wide" and "trading halt" do not occur. "Circuit" appears only for constituent price bands in eligibility rules (for example, pp. 11–12).
- The validator's informational check `first_close_equals_base_value` is **true**. This agrees with the base date and base value on p. 284.

## 8. Spot check (comparison run by the implementer; reference closes from the reviewer)

The 10 dates were drawn by `Generator(PCG64(SeedSequence(20261111)))` (D25). For each date, open the named file at the given line and compare its Close with an independent public source.

| date | file | line | Close matches? | independent source | date checked |
|---|---|---|---|---|---|
| 1997-09-26 | NIFTY 50_Historical_PR_03111996to02111997.csv | 21 | not checked: no independent source found | — | — |
| 1997-10-21 | NIFTY 50_Historical_PR_03111996to02111997.csv | 9 | not checked: no independent source found | — | — |
| 1998-04-13 | NIFTY 50_Historical_PR_03111997to02111998.csv | 141 | not checked: no independent source found | — | — |
| 2002-08-06 | NIFTY 50_Historical_PR_03112001to02112002.csv | 61 | yes (within 0.05 points; RBI rounds to 0.05) | RBI Weekly Statistical Supplement dated 2002-08-17: https://website.rbi.org.in/hi/web/rbi/-/publications/weekly-statistical-supplement/16.-bse-sensitive-index-and-nse-nifty-index-of-ordinary-share-prices-mumbai-5015 | 2026-11-12 |
| 2011-01-12 | NIFTY 50_Historical_PR_03112010to02112011.csv | 202 | yes (within 1.00 points; secondary sources disagree in the last digit) | Goodreturns: https://www.goodreturns.in/news/2011/01/12/bse-sensex-nse-nifty-market-updates-jan-12.html | 2026-11-12 |
| 2017-03-21 | NIFTY 50_Historical_PR_03112016to02112017.csv | 156 | yes (exact) | MoneyWorks4Me: https://www.moneyworks4me.com/company/news/index/id/244220 | 2026-11-12 |
| 2021-12-15 | NIFTY 50_Historical_PR_03112021to02112022.csv | 221 | yes (exact) | 5paisa: https://www.5paisa.com/news/closing-bell-indian-market-falls-for-the-fourth-day-in-a-row | 2026-11-12 |
| 2022-11-23 | NIFTY 50_Historical_PR_03112022to02112023.csv | 236 | yes (exact) | IIFL (indiainfoline): https://www.indiainfoline.com/news/markets/sensex-rises-92-pts-nifty-ends-above-18250-vix-above-14-mark | 2026-11-12 |
| 2026-05-25 | NIFTY 50_Historical_PR_03112025to02112026.csv | 112 | yes (exact) | Oquilia: https://www.oquilia.com/markets/today/2026-05-25 ; Arihant Capital: https://www.arihantcapital.com/News/News-Details/1698699/investor-attention | 2026-11-12 |
| 2026-06-11 | NIFTY 50_Historical_PR_03112025to02112026.csv | 100 | yes (exact) | DSIJ: https://insights.dsij.in/appmindshare/closing-bell-nifty-50-ends-below-23200-sensex-falls-151-points-as-middle-east-tensions-us-inflation-spike-weigh-on-markets-id003-57714 ; Arihant Capital: https://www.arihantcapital.com/News/News-Details/1705104/investor-attention | 2026-11-12 |

Comparison run by the implementer on 2026-11-12 against public reference closes supplied by the reviewer from secondary sources; the owner did not perform it. Coverage 7 of 10; the three 1997–1998 dates are unchecked. The internal check (ohlc_inconsistent = 0) covers all rows with Open/High/Low.

## 9. TRI (Total Return Index)

Until Day 11C the TRI data had not been downloaded. TRI Gross is declared as a sensitivity series in plan §1 (`docs/plan_v2.2.md:348`), D2 (`docs/plan_v2.2.md:727`) and `docs/methodology.md:59`. Its design is D46 (e) and its implementation D48.

**Day 11C (D48).**
- Source: https://www.niftyindices.com/reports/historical-data. Downloaded by the owner on 2026-11-15, unedited, into `data/raw_tri/` (gitignored).
- Files: 28, named `NIFTY 50_Historical_TR_<ddmmyyyy>to<ddmmyyyy>.csv`. Format: quoted, CRLF, newest first. Header `"IndexName","Date","Total Returns Index","Net Total Return Index"`.
- Rows: 6,808. The first date is 1999-06-30, the NSE-documented TR availability start (D46 (e)).
- Only the Gross column ("Total Returns Index") is used. The Net column is ignored: 129 of its cells are "-" and they are never parsed.
- Validation (D25 rules, same `[validation]` constants): 0 hard errors.
- Flags, report only: calendar_gap 12, identical_close_pair 3, large_return 65, stale_close_run 0, weekend_date 32; 365 absent weekdays (D26 audit).
- PR-vs-TRI date-set differences over the whole TRI span: 1 PR-only date, 0 TRI-only dates. Over the OOS span (from the first PR OOS return date): 0 and 0. The dates themselves are listed only in `data/validation_tri/date_differences.csv` (gitignored).
- These counts come from a dry validation run by the D48 code into a scratch directory outside the repository, before the Day-11C-1 commit (D48 (14)). The `validate-tri` command run after that commit must reproduce them.
- Spot check: 10 of 10 dates match niftyindices.com (owner, 2026-11-15). The sheet is `data/validation_tri/spot_check_sheet.txt` (gitignored).

## 10. Dispositions and open items

**Dispositions (D27, approved 2026-11-12):**

1. All 7,717 rows stay in the series as ordinary observations, including the 42 weekend-dated sessions. Nothing is removed, filled or interpolated (methodology §2).
2. The EXPLAINED / UNEXPLAINED / UNVERIFIED labels are documentation only, and no computation uses them. Absent weekdays are not filled.
3. The market-wide halt days (2004-05-17, 2008-01-22, 2009-05-18, 2020-03-13, 2020-03-23) and the 2021-02-24 technical halt keep their official closes. They are recorded as a limitation for the final report: a close pinned by a lower-circuit halt could understate that day's loss. No adjustment is made.
4. The 1997-10-02 to 1997-10-08 suspension is one return spanning the gap, as under methodology §2.
5. The sample size is computed from the data: 7,716 returns and an OOS size of 5,716. Kupiec regions, the Newey–West lag and the bootstrap block length are computed at the actual T. The final partial traffic-light block has 216 observations and gets no zone. The T = 5,650 values in `docs/reference_values.py` remain unit-test fixtures, and the plan is not edited.
6. TRI Gross is obtained before the declared sensitivities (Day 11).

**Still open:**

- The terms of use were not re-verified: the fetch was blocked or timed out (§1).
- The EXPLAINED labels rest on search-result summaries; the cited pages were not opened (§5).
- Three spot-check dates (1997-09-26, 1997-10-21, 1998-04-13) are unchecked (§8).
