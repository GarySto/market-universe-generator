"""
backfill_tickers.py — Fills tickers.txt up to 470 in-range momentum candidates.

Filters applied:
  - Price $1.50 to $75
  - Average daily volume above 500,000 (raised from 200k for reliable OTC fills)
  - Excludes known ETFs, index funds, and non-stock instruments
  - Excludes tickers with fund-like suffixes (preferred shares, warrants etc)
  - US common stocks only

Run from your market-universe-generator folder:
    py backfill_tickers.py

Requires: tickers_cleaned.txt (from refresh_tickers.py)
Output:   tickers_final.txt
"""

import yfinance as yf
import pandas as pd
import time
import os
import re

TARGET     = 470
PRICE_MIN  = 1.50
PRICE_MAX  = 75.0
VOL_MIN    = 500_000   # raised — need reliable OTC fills in premarket
BATCH_SIZE = 50

# ── Known ETFs and funds to exclude ──────────────────────────────────────────
EXCLUDED_TICKERS = {
    # Major ETFs
    "SPY","QQQ","IWM","DIA","GLD","SLV","TLT","HYG","LQD","XLF","XLE","XLK",
    "XLV","XLI","XLU","XLP","XLB","XLY","XLC","XLRE","VTI","VOO","IVV","VEA",
    "VWO","EFA","EEM","AGG","BND","ARKK","ARKG","ARKW","ARKF","ARKQ","ARKX",
    "SQQQ","TQQQ","SPXS","SPXL","UVXY","SVXY","VXX","VIXY","SOXL","SOXS",
    "LABU","LABD","TECL","TECS","NAIL","DRN","DRV","CURE","WANT","HIBL","HIBS",
    "FNGU","FNGD","WEBL","WEBS","DFEN","DUSL","RETL","VIRT","MIDU","SMDD",
    "UDOW","SDOW","URTY","SRTY","USMV","QUAL","MTUM","VLUE","SIZE","EFAV",
    "IEMG","ITOT","IXUS","SCHB","SCHA","SCHM","SCHD","SCHG","SCHV","SCHF",
    "SCHE","SCHP","SCHI","SCHR","SCHH","SCHZ","SCHQ","SCHC","SCHO",
    # Sector ETFs
    "GDX","GDXJ","SIL","SILJ","REMX","LIT","ICLN","TAN","FAN","QCLN","ACES",
    "JETS","AWAY","BETZ","VICE","HERO","ESPO","NERD","GAMR","CLOU","SKYY",
    "HACK","BUG","CIBR","IHAK","WCLD","IGV","SOXX","FTEC","VGT","IYW","QTEC",
    "FDN","PNQI","IBUY","ONLN","MOTO","DRIV","KARS","IDRV","CARZ","VCAR",
    "ROBO","BOTZ","IRBO","THNQ","DTEC","LRNZ","EDUT","SOCL","KWEB","CQQQ",
    "MCHI","CHIQ","CHIE","CHIX","CHIE","ASHR","ASHS","KBA","FXI","YINN","YANG",
    # Bond ETFs
    "SHY","IEF","IEI","TIP","VTIP","STIP","LTPZ","TIPX","SCHP","IPE",
    "BKLN","SRLN","PFLD","PFXF","JNK","SJNK","USHY","FALN","ANGL","HYS",
    # Leveraged/inverse
    "SSO","SDS","UPRO","SPXU","QLD","QID","TQQQ","SQQQ","DDM","DXD",
    "UWM","TWM","MVV","MZZ","SAA","SDD","UKK","SKK","ROM","REW","UGE","SZK",
    "UCC","SCC","RXL","RXD","UYG","SKF","UYM","SMN","URE","SRS","USD","SSG",
}

# ── Ticker pattern exclusions ─────────────────────────────────────────────────
def is_excluded_by_pattern(ticker):
    # Preferred shares (end in P, PA, PB, PC, PD, PE)
    if re.match(r'^[A-Z]+P[A-E]?$', ticker) and len(ticker) > 3:
        return True
    # Warrants (end in W or WS)
    if ticker.endswith('W') or ticker.endswith('WS'):
        return True
    # Rights (end in R)
    if ticker.endswith('R') and len(ticker) > 2:
        return True
    # Units (end in U)
    if ticker.endswith('U') and len(ticker) > 2:
        return True
    # Very long tickers (5+ chars often indicate special instruments)
    if len(ticker) > 5:
        return True
    # Numbers in ticker (usually warrants or units)
    if any(c.isdigit() for c in ticker):
        return True
    return False

T212_US_TICKERS = [
    "STN", "JZ", "SGD", "STRA", "LTRPB", "HBT", "BRKR", "AKAN", "AUPH", "ALVR",
    "DYAI", "AAPL", "LLYVK", "CETY", "MSFT", "ROAD", "EVVTY", "HOFV", "RGTPQ", "CARR",
    "CPS", "NTWK", "HLIT", "VOD", "UUU", "RXRX", "CYCC", "HCAT", "ELME", "NYT",
    "NOW", "IFF", "RLX", "NRGV", "JOBY", "IKT", "MGNX", "ZION", "AMZN", "PDD",
    "MINM", "PSMT", "PHIN", "ISRL", "VKTX", "ITW", "NVGI", "HCP", "RCMT", "AZREF",
    "OPCH", "SNBR", "CAVA", "EDN", "ARQQ", "FROG", "UBER", "LLY", "SMG", "GLOB",
    "VLGEA", "FLNT", "VWDRY", "CIEN", "DGX", "JSDA", "FDMT", "PLX", "RVP", "EH",
    "LINE", "IE", "EMCG", "BHP", "IPHA", "ATO", "BCC", "AEP", "LPL", "PALI",
    "JXN", "NNAG", "UPMMY", "GREE", "AGM", "DGII", "OESX", "ICUI", "SWAG", "NXE",
    "FEAM", "SQFT", "CALC", "DOYU", "DVA", "LILAK", "LFST", "MSAI", "PRCT", "BUR",
    "DQ", "CROX", "DBRG", "BNED", "GXO", "SRUUF", "FIP", "TUP", "CHMI", "CLSK",
    "ATS", "RSLS", "AHG", "BA", "HUMA", "SM", "INFU", "COSM", "VRT", "ONTF",
    "PRZO", "ABT", "STLD", "CYGIY", "RRR", "VOXR", "TSPH", "HCWB", "WOLF", "CAAP",
    "VMEO", "SENR", "Z", "UA", "DAC", "ATMU", "ISUNQ", "BBAI", "SMX", "IMPP",
    "BWA", "TBN", "ODD", "PRLB", "SGBX", "AREC", "OSW", "FLD", "JDZG", "WY",
    "CHE", "RGNX", "IEX", "PHYT", "LUV", "PIRS", "SLRC", "AACT", "NRBO", "LDWY",
    "BXRXQ", "SON", "GBR", "FQVLF", "YPF", "ELEV", "PLAY", "EFOI", "MRNS", "AVB",
    "HIVE", "DOOO", "SPRB", "IBRX", "MRTN", "ENLC", "TKLF", "BOC", "SYNA", "DCI",
    "PPTA", "GWRS", "SFTBY", "EDTK", "TMC", "CNX", "OS", "IFBD", "PCSC", "WBS",
    "ADD", "CRYM", "FG", "ALUR", "FTHWF", "BMRC", "EBR", "ENZ", "F", "CHGG",
    "LVS", "CGON", "STX", "CERO", "ASLE", "IMRN", "INAB", "RXO", "GLNG", "HUT",
    "M", "TNDM", "ADXN", "NHYDY", "ACM", "FAST", "AGMH", "VVOS", "CSR", "NISN",
    "PERI", "BCLI", "ASND", "LIFD", "ASC", "HRMY", "ALK", "FRBA", "CERT", "BIOR",
    "GOTU", "SITC", "NUWE", "LAW", "LUCD", "KROS", "TTOO", "BGXX", "EXPE", "MDIA",
    "KZR", "DDC", "CAR", "LYV", "AMWL", "HYMC", "ADVM", "OSCR", "CRCT", "EC",
    "TROX", "SMMNY", "KEWL", "MPTI", "KNOP", "PUBM", "DTI", "TAOP", "IGT", "CABA",
    "VSTM", "BALY", "ZVOI", "NATL", "INKT", "GNCAQ", "AQST", "SKYE", "CHX", "NVOS",
    "AGRX", "BWLP", "NESR", "ONON", "SRNE", "S", "TDY", "FLEX", "CMRA", "LDOS",
    "NL", "EVTL", "CEAD", "LADX", "IPDN", "SPRS", "ENLAY", "PED", "GRBK", "PLD",
    "NEM", "CLOV", "REEMF", "ACIC", "EMN", "HR", "ALT", "WAY", "MAMO", "AMBO",
    "SFL", "KRNT", "HDSN", "TOBAF", "QD", "TOI", "LANC", "BYFC", "ASGLY", "APOG",
    "MPLN", "ZI", "RENB", "ENTG", "TBIO", "EVFM", "CNI", "BRZE", "XELA", "PACS",
    "RARE", "ALPP", "NBY", "STAF", "SSD", "WIT", "YVRLF", "OBLG", "TOL", "GNL",
    "CVS", "ABCL", "GDEV", "SUNS", "QDEL", "LVTX", "SRZN", "CTMX", "HGENQ", "VICR",
    "BLX", "CRDL", "RYCEY", "KIRK", "ATKR", "TYIDY", "LITOF", "XENE", "ASNS", "PLAG",
    "PANW", "BXSL", "STAB", "IMO", "CRT", "VRPX", "PALT", "AYTU", "CTNT", "FAT",
    "SOUN", "ACTHF", "SILA", "CPRI", "VRTX", "LGO", "ZETA", "NEE", "BDN", "HLP",
    "FTCO", "HBI", "VTYX", "BAP", "STAG", "TFC", "MCO", "NRDS", "APLS", "LTBR",
    "MANU", "QCOM", "J", "SUPN", "KNSL", "ACLX", "BLCM", "CISS", "HOLO", "SHEN",
    "LQDA", "WIMI", "SRBK", "CELZ", "VXRT", "LLKKF", "PPG", "LAND", "WVE", "GIL",
    "ALLK", "HNNMY", "HIG", "MLGO", "FTV", "CGEM", "AADI", "CDTG", "JYD", "OP",
    "LQDT", "APTV", "HELE", "AX", "SKYW", "ACHR", "AG", "RGC", "GLUE", "ATRA",
    "CNQ", "COE", "RYES", "ARC", "WTW", "SANM", "JTKWY", "SPRU", "VAL", "GPC",
    "TMDX", "NXT", "NCSM", "SG", "LRHC", "COP", "CBWTF", "RIOCF", "CNET", "SPHR",
    "PHAT", "WRAP", "GPI", "MHO", "GGAL", "EVOL", "SPXCY", "APPS", "ETN", "SAXPY",
    "ILLMF", "ACAB", "GMED", "ALL", "RCKT", "GCTS", "VHC", "ROP", "IMTE", "CINF",
    "COHR", "GEHC", "OBIO", "KREF", "MUSA", "VANI", "HAO", "EHAB", "RGLS", "FITB",
    "VSH", "ATEN", "TGB", "CR", "KFY", "AGI", "TXN", "NTNX", "TREX", "SABR",
    "STIXF", "BITF", "SNTG", "GCO", "IAG", "BRAXF", "BWAY", "YMM", "ALFIQ", "REVG",
    "QFIN", "MGLD", "KULR", "BFLY", "BMI", "ZNOG", "PDYN", "ASAN", "LVO", "WHR",
    "TRMD", "MXL", "SGLY", "TDOC", "MRKR", "GRDI", "LTRPA", "SHW", "SVCO", "RBRK",
    "PLTR", "SFBS", "ZAPP", "META", "GOGO", "TCNNF", "ANGPY", "BLRX", "IBP", "HTGMQ",
    "CTKYY", "DMGGF", "ES", "TARS", "CAN", "XTKG", "GRYP", "LXU", "GDS", "HCANF",
    "VRAR", "SLGL", "AUNA", "FTII", "GNTX", "FTS", "INSG", "RBA", "ALLG", "ICD",
    "ECIA", "CTRA", "PD", "DNNGY", "MMC", "MASS", "MNMD", "GALT", "DARE", "ZBAO",
    "PMEC", "PWR", "FLGC", "WLGS", "HOOD", "MSWV", "AGEN", "PRPO", "SSL", "KIND",
    "JWN", "RPRX", "KRG", "GPCR", "DM", "LODE", "WELL", "INBS", "LYRA", "BRAC",
    "SWI", "OPT", "BHM", "MAIA", "CPTN", "CADL", "RADCQ", "ROIV", "GEO", "NTR",
    "LPG", "DHAI", "SEEL", "GILD", "ADV", "INFIQ", "TOGI", "AHT", "KA", "MODG",
    "ELOX", "SGIOY", "AMPS", "ASTH", "GILT", "UROY", "ADP", "CTXR", "AES", "LICY",
    "YEWB", "PTC", "GDHG", "PI", "CRDF", "MNDY", "FOLD", "IBTA", "GRND", "HLIO",
    "SOHU", "XYL", "OM", "ALOT", "CJET", "CELU", "CNSP", "VTVT", "SINT", "GPOR",
    "GLAC", "LGMK", "DXF", "DDS", "PRSO", "BLDR", "BLDP", "AMRK", "IVCA", "PBT",
    "CRMT", "PGY", "KOD", "ATXI", "NFG", "GVP", "DLO", "MICS", "SHMD", "ANVS",
    "SEZL", "DFH", "MEDP", "SCYX", "SEAC", "RFIL", "IHRT", "UHAL", "TITN", "MHUA",
    "PLCE", "INBX", "PSTX", "CFG", "VVPR", "INTC", "NGG", "IDXX", "TIGR", "ASOMY",
    "LYB", "ALDX", "MOGO", "SILC", "PWM", "SKIL", "D", "DINO", "BOW", "DOMH",
    "RPM", "BEN", "GLTO", "GZPFY", "AUVIQ", "IIIN", "QRTEA", "ETAOF", "MCFT", "RVLPQ",
    "DOC", "WAFU", "AXL", "CHNR", "BB", "NTZ", "RDW", "UHS", "FRHC", "CMTL",
    "DEA", "UEC", "WEIDY", "ATHXQ", "CNTB", "JCI", "UGP", "REXR", "AZN", "TU",
    "SIGI", "NEXA", "ELYS", "TLSA", "MSM", "CRXTQ", "CLLS", "ZS", "MARX", "IPW",
    "FINMY", "ZYXI", "MGIH", "ENB", "KUKE", "ETWO", "LEG", "EMBC", "CMA", "BIMI",
    "AZO", "APO", "WST", "INMB", "SLB", "ARCH", "ASST", "MRT", "PRFX", "KMTUY",
    "SALM", "CWBR", "PEGY", "BVN", "MAIN", "UPST", "MKULQ", "RIO", "LUVU", "WBX",
    "RR", "PRLD", "SOPA", "AMP", "CNTGF", "RMNI", "NTCT", "LILM", "CHPT", "FICO",
    "PDCO", "EGOXF", "ROCK", "EVTV", "BETSF", "RECT", "SPCB", "SJT", "KRYS", "MRM",
    "TTEC", "MOTS", "CCS", "ALGN", "SPOT", "NC", "ARBE", "LAES", "PRKS", "GES",
    "DMAC", "QIWI", "CASY", "FCUUF", "LIXT", "SBNY", "BANL", "NTRA", "SRAX", "AZZ",
    "KFRC", "PINS", "VIAV", "GSK", "ONDS", "CNFR", "CRVL", "GOLLQ", "MATH", "SMXT",
    "MSGM", "NVT", "RAASY", "PNR", "MGA", "MEI", "BEAT", "UDMY", "SQBGQ", "HOG",
    "YJ", "ACET", "MBIO", "SNY", "EYPT", "ENG", "FENC", "HIFS", "FTAI", "SAFE",
    "CRH", "MLEC", "BCYC", "SWN", "NNVC", "KO", "SNOA", "PRG", "AR", "CPSH",
    "GTN", "MOND", "BHVN", "LNZA", "CYBN", "ADIL", "BRBR", "HWNI", "CNP", "ALIM",
    "TER", "LMDXF", "ZURA", "GNPX", "ASO", "UONE", "FCPT", "NNE", "OWLT", "PZG",
    "ABEV", "LFLY", "PHCG", "ESAIY", "AMLI", "TCEHY", "IBIO", "TDG", "BRCNF", "FPAY",
    "NSPR", "STIM", "QURE", "RHI", "BNRG", "ALCY", "BMY", "DPRO", "CVE", "DLAKY",
    "MOBQ", "SNWGF", "CELH", "FR", "GAME", "FVRR", "TCON", "BFH", "AY", "CB",
    "ADMA", "NU", "GAQ", "TMRC", "GWAV", "WEC", "GROY", "YALA", "BKR", "DT",
    "KORE", "MTA", "OGN", "LUNA", "NOG", "AATC", "HIHO", "GLBS", "NTAP", "KWBT",
    "VTGN", "ATYR", "HUDA", "VEV", "MET", "TEX", "BAH", "CHRS", "O", "ALRM",
    "ZYME", "CDIO", "III", "LGHL", "EARN", "SLP", "MDJH", "MKC", "EEIQ", "AVTBF",
    "ETSY", "EQX", "BOX", "GROV", "VRME", "LGDTF", "NRDE", "NLST", "HROW", "ERO",
    "CRVS", "XIACY", "ERAS", "REED", "EVGO", "LW", "CNL", "AEVA", "NTBL", "GAN",
    "A", "PLSE", "IZEA", "FTDR", "LPTX", "LBTI", "CLWT", "FSP", "MAG", "AMST",
    "VWAPY", "UBCP", "SAGE", "KLG", "SHOT", "ACRV", "TREE", "HUBS", "AMLX", "LBPH",
    "AMRC", "ORGS", "POAI", "OCGN", "BLMH", "SCWX", "QRTEB", "SENS", "DELL", "NAT",
    "ROK", "SER", "CAAS", "RRX", "GGEI", "EPSN", "GOCO", "CDW", "LOGC", "MSS",
    "ALV", "SJW", "BPT", "TEF", "KIROY", "C", "PLYA", "TERN", "AFJKU", "RIBT",
    "ABUS", "TENX", "OZON", "PSTG", "SSYS", "TUYA", "PET", "BC", "JZXN", "BBLNF",
    "GMRE", "TRV", "UMH", "FF", "MTCR", "IVDA", "NTRB", "MPU", "NVEC", "HRZN",
    "WPC", "CLDT", "NXTT", "FTI", "EU", "BYON", "LFT", "AIU", "HXL", "WYNN",
    "AKBA", "GANX", "YNDX", "RVSN", "KEN", "PPL", "HEI", "THO", "BRY", "CAL",
    "GTCH", "PNC", "VRNOF", "MGTX", "MDWD", "LIF", "SNX", "IQ", "PMN", "QXO",
    "JSPR", "GLW", "VC", "VERV", "BLBX", "KITT", "GE", "OHI", "VRDN", "GGR",
    "REZZF", "CGRNQ", "VINC", "YHGJ", "NVFY", "SYF", "XELB", "DFS", "ALZN", "PODD",
    "MLI", "WHD", "HOLX", "EQOSQ", "KNW", "HSAI", "HWM", "NURO", "PHCFF", "BKD",
    "NRXS", "ZTO", "SLNAF", "SUWN", "PAY", "OKYO", "EE", "ZBRA", "NEON", "STBX",
    "LEU", "ISPR", "PG", "NMRA", "GOLD", "FACO", "NGLOY", "KOF", "OPRA", "VIR",
    "TIL", "CCIX", "MGIC", "GTBIF", "TPL", "OTEX", "ADXS", "FPI", "BVNRY", "DAVE",
    "ICU", "ARVN", "AL", "TTE", "SJM", "PGR", "MONRY", "EDU", "CCJ", "MUR",
    "PTIX", "HWH", "API", "WFC", "BBKCF", "LSPD", "PLBY", "OREAF", "DADA", "AHH",
    "TAUG", "NTGR", "UP", "UPXI", "EXPD", "CWAN", "EVRG", "TSHA", "MCHP", "LUMN",
    "AUDC", "SRFM", "GM", "WVVI", "MRO", "EQNR", "TRTX", "HMY", "LGIQ", "INCY",
    "BODI", "NUTX", "SOGP", "CXM", "SAVE", "SBIG", "CHR", "ABLV", "ARAY", "TS",
    "OR", "VIOT", "IDT", "IMPUY", "BAOS", "EFC", "SCL", "CGBS", "CPRT", "SPMC",
    "ENSV", "UGRO", "ALLO", "SNSE", "SDCCQ", "CDT", "PAC", "BIDU", "CWD", "WWW",
    "FOX", "ESPR", "INUV", "CCCC", "CRIS", "MNTK", "AEMD", "CHCI", "IPXX", "TBMC",
    "COIN", "ZPTA", "LITE", "INIS", "UPS", "IRTC", "GETY", "DAN", "ELTP", "ITP",
    "ATIP", "OTIS", "AMPL", "FSHP", "PR", "SEIC", "ESRT", "EXK", "DOGZ", "NEPTF",
    "EGHT", "EXR", "DRS", "EXTO", "YARIY", "HITI", "BDSX", "ACORQ", "KALA", "APCX",
    "GBNHF", "XHR", "WW", "TNK", "PARA", "MREO", "MAQC", "ACA", "SCCO", "SOFI",
    "ZIVO", "WOOF", "ISRG", "JYNT", "DEFTF", "AVPT", "SMRT", "UTMD", "INDP", "SMCI",
    "HTZ", "CRON", "HNST", "APRE", "ZWS", "OLED", "ATOM", "NXL", "SSTK", "JL",
    "PNPNF", "POWI", "PZZA", "GD", "EGIEY", "DOMO", "ENS", "WIX", "MKKGY", "NFLX",
    "AWK", "RNG", "ARW", "GMM", "SNGX", "OKTA", "MESO", "MCY", "INHD", "HEPA",
    "MELI", "LWLG", "LBTYA", "PTPI", "TWOUQ", "DNLI", "JUSHF", "KC", "NABL", "CNC",
    "CPIX", "PTSI", "VGR", "VOXX", "JHG", "MMSI", "AMRX", "WTER", "PGTK", "JOE",
    "QSI", "XYF", "H", "DHR", "ENSC", "HUDI", "AGRO", "PENN", "ZGN", "PFMT",
    "DDOG", "HES", "BMR", "VRM", "PAGS", "CAG", "WEJOF", "ESTC", "ICMB", "CWT",
    "CLSD", "AGILQ", "ALSN", "CTVA", "PCT", "EEFT", "OII", "MNPR", "ABTS", "APT",
    "XPEV", "CRWD", "OPTX", "SLM", "GRAL", "IBKR", "BEEM", "LMND", "BNTC", "EFXT",
    "ONCO", "TPIC", "CRTD", "FL", "MSDL", "ENTX", "DVN", "ABBRF", "TRP", "DOCU",
    "MLCO", "MDBH", "AMKR", "UGI", "GRVY", "VMAR", "TRSG", "BAK", "LSDIF", "INM",
    "MASI", "NEO", "NET", "TWO", "CAJPY", "KBNT", "REAL", "SPCE", "LOB", "ZVIA",
    "EVC", "AUST", "BGI", "VNTRF", "XRX", "NTOIY", "QS", "UFPI", "KOPN", "TDC",
    "EQIX", "SIMO", "AFRI", "VIRT", "DXLG", "ANSC", "NKLA", "DOCN", "FEDU", "TOITF",
    "HGTY", "AJX", "ACR", "IREN", "STRW", "SKIN", "NWS", "SURG", "YOSH", "PTALF",
    "AFRM", "CM", "MKTX", "NDRA", "NTRP", "FRSH", "INGR", "PVH", "ADRNY", "MNDO",
    "ACDC", "EBON", "NCI", "BGLC", "SRG", "MOS", "HLTHQ", "OZK", "SD", "AZEK",
    "ABM", "VLN", "MUFG", "ONDR", "CLIS", "RENT", "RHHBY", "PII", "TCJH", "DAL",
    "KMB", "REVB", "MSN", "ANRO", "AUMN", "SATX", "GFI", "MHK", "SRM", "NTES",
    "MRNA", "LOMA", "TUEMQ", "BEST", "HTHT", "CMG", "KARO", "CRBG", "HUSA", "CARM",
    "DKS", "CEP", "JNPR", "ANTE", "ALE", "ONVO", "APLM", "ALAR", "CMRX", "VCIG",
    "SO", "TH", "ACB", "TTI", "APWC", "BLAC", "GAMB", "CPT", "SOL", "UMC",
    "EFTR", "VIST", "COR", "SEDG", "CVAC", "FORM", "GLYC", "VEON", "POLCQ", "FUL",
    "DBGI", "TLIS", "LSF", "ALLT", "BLFS", "OCFT", "JWEL", "SAN", "FBYD", "SNDL",
    "LIPO", "SRCE", "VNET", "AMAT", "RMD", "NXPI", "OMER", "CRC", "UPLD", "TXT",
    "CTO", "NVA", "GENK", "TARA", "RDIB", "LAAC", "AREB", "MITT", "ADI", "BQ",
    "AVY", "U", "UPC", "SWK", "SCS", "CXAI", "BTMD", "SONM", "ASXC", "FN",
    "ELWS", "VEEV", "MSA", "LNC", "MSTR", "GNLN", "ALTO", "NVCT", "HLN", "TMDIF",
    "CVV", "TTNP", "GAXYQ", "SVMH", "CODX", "GFS", "QMMM", "LC", "NOC", "OGE",
    "IGMS", "MATX", "AAP", "CLIR", "EBAY", "CAMT", "CBRE", "IMCC", "WK", "FBRT",
    "ZVRA", "FMCC", "URGN", "UBSI", "COYA", "ADEA", "MGNI", "LFVN", "AIM", "VS",
    "VLCN", "HSBC", "KEY", "GDEN", "PGNY", "AUTL", "PT", "OPXS", "TIRX", "DUK",
    "VERX", "DAWN", "OUST", "OMGA", "VLVLY", "ATLN", "PX", "GGAAF", "XP", "ELVA",
    "CVX", "NRXP", "GTBP", "TPCS", "STRR", "CHEK", "LLAP", "CCO", "TWG", "EVRI",
    "GBCI", "WDH", "CBUS", "DOLE", "RCL", "FDUS", "COCP", "TSE", "WDS", "XPO",
    "CMPS", "RNW", "FULC", "AMH", "JRNGF", "MRAI", "UBXG", "NMIH", "ARE", "QUIK",
    "LBRT", "REGN", "PAR", "FATE", "SICP", "PRVA", "ALPMY", "FEBO", "CLVR", "DECK",
    "DLTR", "GXAI", "XTIA", "FGF", "STKS", "JVA", "MLKN", "NEWYY", "HUBB", "ORMP",
    "KEYS", "PAVM", "HVT", "VSTE", "TEAM", "COUR", "SPRC", "BABA", "KNYJY", "BILI",
    "RIGL", "AN", "SNAP", "NVIVQ", "FI", "NKTX", "SNAL", "X", "IMOS", "VZLA",
    "TME", "IBN", "SB", "EJH", "ITT", "CRUS", "IMMP", "ORINY", "PNBK", "AETUF",
    "DAVA", "AGR", "CXW", "MACI", "ENLT", "NVEI", "SGE", "ZJYL", "GERN", "BPTH",
    "CX", "GOFPY", "SQ", "RBOT", "OLK", "KSCP", "ANET", "FHSEY", "VVX", "ELMSQ",
    "BRSHF", "RDUS", "MS", "SEAT", "TXRH", "CAUD", "XOS", "WT", "INFN", "LUCY",
    "OUKPY", "PXMD", "BMRA", "VMD", "SEER", "SKT", "SLHGF", "IONQ", "ILMN", "EVGN",
    "OLMA", "THMO", "POWL", "EQ", "PEPG", "TSBX", "AFL", "SLG", "GROM", "MKDTY",
    "LPSN", "VIRX", "SSY", "VIRI", "INDI", "PASG", "QUBT", "NSSC", "GECC", "FUN",
    "LNW", "IBM", "REZI", "BKKT", "GBDC", "RS", "BBDC", "PXDT", "ORAN", "AYI",
    "PRAX", "IOT", "UK", "VLTO", "XWEL", "ELYM", "ASAZY", "TPET", "CVM", "ADAP",
    "RXT", "SRPT", "TTC", "GPRO", "DOW", "TRAK", "VISL", "RDFN", "ERIC", "MNKD",
    "OPFI", "OTCM", "EBS", "ABL", "EDRY", "CART", "KBR", "SHPH", "LBTYB", "ROMA",
    "ITRM", "EDIT", "BMO", "RGS", "JBLU", "AMT", "ELDN", "OTGLY", "MSB", "DPZ",
    "EMWPF", "OVV", "GEVO", "CMND", "GWH", "ANY", "JAGX", "TURB", "APPF", "CMI",
    "MYSZ", "GPRE", "DNA", "SGMO", "CVGW", "UNIT", "TLSS", "MGRM", "CYDY", "ICCM",
    "OLP", "BGC", "GOGL", "MXC", "SBRA", "KN", "MEGL", "CMP", "ALLE", "BLPH",
    "SSB", "ATRO", "ARI", "CGNT", "INVU", "TPHS", "CDLX", "EVCM", "VSAT", "WGO",
    "CGEN", "LPTV", "SLF", "CEIX", "RMBL", "NVS", "BBIG", "IMNN", "LIANY", "NRDY",
    "STAA", "ORCL", "PM", "SHYF", "MNDR", "CRSP", "WMS", "SPTN", "VAXX", "AVTX",
    "SHIP", "VWAGY", "CCTG", "FGI", "GUTS", "FUBO", "XTNT", "WMT", "TTGT", "IMUX",
    "LVWR", "LNT", "ITOCY", "AXTI", "SPGI", "AMED", "BIRD", "WINT", "LAC", "KSPI",
    "FLYE", "DJCO", "WRB", "MBT", "ERII", "CNA", "GRI", "PEN", "TRUP", "ARAV",
    "ODTC", "OPEN", "ADAL", "REKR", "JKHY", "SGMA", "BBD", "NYCB", "SLCA", "AESI",
    "TEVA", "RWAY", "ESHA", "SND", "TLN", "OMC", "OCG", "TGS", "MSC", "IONS",
    "SLI", "PLNH", "URG", "HRYU", "NTIC", "CFLT", "DO", "DBI", "ELTX", "SNOW",
    "EW", "MOGU", "MLTX", "GTE", "VBIVQ", "FKWL", "ATPC", "DAR", "GPN", "VICI",
    "LADR", "CSIQ", "BREA", "THAR", "VYGR", "UUUU", "KLAC", "NGS", "AMPY", "COMP",
    "RSTN", "FTFT", "ANIX", "TALO", "HCTI", "SEG", "CTOR", "URBN", "DVAX", "TCOM",
    "STMH", "KRKNF", "STG", "ZM", "FRGE", "OFG", "NERV", "SHFS", "TOON", "COO",
    "BLBD", "FTLF", "YSG", "RACE", "LAKE", "BHR", "XFOR", "LPCN", "TM", "SHRG",
    "ULS", "DDD", "VYNT", "RCKTF", "TTM", "TVTX", "CVII", "MODD", "FRTX", "FRT",
    "REE", "MYO", "RELL", "CSX", "PFSI", "TSVT", "SHPWQ", "MDGL", "FOA", "LSB",
    "FLGT", "MPC", "FNB", "QRVO", "CWBHF", "INSW", "VNDA", "LEV", "EVKG", "PBLA",
    "MMM", "EMX", "RVYL", "HCDIQ", "PURE", "VGFCQ", "PLUG", "OVID", "VFF", "KLXE",
    "MAA", "DRIO", "PYPL", "LUNR", "KBDC", "ZKIN", "PRME", "BWXT", "RSG", "SVRA",
    "VIG", "INSE", "SGML", "SBAC", "SPT", "CC", "KTRA", "SMR", "SBR", "CAPC",
    "EOSE", "ABEPF", "FNMA", "SID", "BFI", "ROLR", "PRPL", "GODN", "NBEVQ", "ALRN",
    "FFIE", "VTSI", "BTCT", "TCBP", "VSAC", "BYNO", "ETCMY", "LENZ", "GTEC", "KTTA",
    "AGAE", "CSSEQ", "GOOGL", "SW", "WDFC", "CCI", "MP", "OTLY", "BLNK", "RUN",
    "SMTC", "SACH", "SLDP", "GSL", "LXRX", "GFF", "GIPR", "KR", "PNNT", "MKTW",
    "BRFH", "MTDR", "SYT", "FIVN", "VLOWY", "FTRE", "VOC", "TBLT", "MITQ", "NVNO",
    "FLMMF", "SLNG", "ASTS", "WAT", "AMPE", "ULBI", "RTC", "SAVA", "AMLM", "MDVLQ",
    "CCM", "CRMD", "AVAV", "APTX", "OCUP", "CCL", "FWONA", "ATAT", "BYU", "UNF",
    "LSCC", "MQ", "SAND", "IMVIF", "CFB", "AS", "AER", "BIGC", "SAI", "FLNG",
    "LOVE", "ASTC", "ATXS", "YZCAY", "AM", "USGO", "BLDE", "AUCOY", "NNDM", "CMBT",
    "SKWD", "JPC", "DUOT", "SXT", "EBET", "LSTA", "PIXY", "NEWT", "KDLY", "ABR",
    "FNV", "MIRA", "GMAB", "GHSI", "PATH", "PHM", "OGEN", "KRBP", "TLS", "CMAX",
    "IRT", "LTC", "AWR", "LYG", "IOBT", "VSCO", "ABSI", "JAKK", "PLRX", "SXTP",
    "XCUR", "BTDR", "SCHL", "SCM", "GFASY", "CCRN", "INGN", "FOJCY", "GME", "HCMC",
    "PLAB", "HIMS", "ONTO", "SRRK", "VERO", "GV", "BIG", "OMQS", "PCAR", "UNM",
    "BDRX", "ACON", "FFIV", "LUNMF", "HLT", "LESL", "BYND", "JNVR", "RRC", "TV",
    "ACLLY", "NBSE", "OCUL", "MURA", "TXG", "VALE", "BAM", "RKLB", "TASK", "GRC",
    "PCSA", "AAOI", "MSGE", "GDYN", "KNTK", "MFA", "GSPE", "CLS", "POAHY", "DXR",
    "G", "FNF", "SMTK", "AMSC", "MA", "PHGE", "CGA", "PGHL", "OLB", "PRPH",
    "INTJ", "IONM", "CLPS", "NYMXF", "NNN", "STWD", "NRT", "EGY", "EPAC", "LOAR",
    "WRN", "GO", "HLGN", "GMVHY", "BN", "HTCR", "RELI", "TZOO", "CMRE", "DIDIY",
    "ADAG", "SDIG", "MGY", "NITO", "NDSN", "WWR", "BKHA", "LRE", "IMRX", "SPRO",
    "CNVS", "AOSL", "RSKD", "TCRX", "OGBLY", "HBAN", "BXC", "INFY", "SYSX", "ELS",
    "VNO", "UNAM", "CMMB", "BRX", "JBGS", "XBIT", "BLZE", "IKNA", "MNOV", "REPX",
    "UI", "FMC", "LGFRY", "NSYS", "AAGIY", "ACGL", "UPHL", "UNFI", "ATHA", "SFIX",
    "FORL", "EPOW", "IVST", "MRMD", "IVFH", "NTLA", "OCSL", "ARRY", "MBRX", "SCPX",
    "NCTY", "HGBL", "CNEY", "TANH", "GRTS", "ORA", "MU", "CALT", "ECOR", "PNW",
    "CGNX", "TMVWY", "NEUE", "AMSWA", "BXMT", "CUE", "ARWR", "IHT", "COTY", "DG",
    "SNT", "GTX", "UTZ", "IONR", "CMGR", "GULRY", "MRVI", "BCS", "UAVS", "HAYW",
    "TYRA", "CYTH", "BIOX", "BKNG", "BTOG", "ABEO", "BZUN", "TAK", "ZIP", "ADYEY",
    "NINE", "ELBM", "ARREF", "NVO", "MAMA", "HRB", "OPGN", "ZG", "MEDS", "XPER",
    "LMB", "SCSC", "ALTM", "INDB", "GNK", "CURV", "AYRO", "BLUE", "LFMD", "ARMN",
    "BDORY", "GORV", "OSTX", "JEF", "CRLBF", "IINN", "MLSS", "AEG", "FSLY", "FTCI",
    "AMPG", "STM", "MKL", "SPIR", "NREF", "TTD", "CCEP", "EXAI", "BR", "HWAL",
    "CNTX", "SMAR", "QTRX", "PDSB", "MBUU", "DSX", "QLGN", "EXC", "AGX", "AQMS",
    "NE", "BG", "GYRE", "ACTU", "AHI", "GCI", "MITK", "RFAI", "CSCI", "ROII",
    "TSN", "DTC", "BARK", "ELUT", "APLD", "ELTK", "GLSHQ", "QNCX", "EBTC", "UXIN",
    "MKDW", "TOELY", "ABOS", "VRSK", "HY", "SMFG", "RAMP", "EVOK", "XAIR", "WULF",
    "STCN", "ISENF", "NBBK", "DJT", "ACMR", "CHD", "HAL", "EVA", "AEYE", "MRAM",
    "IIPR", "W", "ALKS", "ARLO", "UL", "BIOL", "WKEY", "RNLX", "FNGR", "WFRD",
    "MGM", "TWFG", "ALEC", "BLK", "MZDAY", "LI", "THRY", "MFC", "CYBR", "AXON",
    "VIK", "FEMY", "IOVA", "FTEK", "SEOVF", "SKLZ", "DC", "SPG", "PRCX", "CLOE",
    "GFAI", "PL", "GDDY", "DMTKQ", "SNTI", "SAP", "ASRT", "ALB", "MAPS", "RF",
    "BAND", "FHN", "GMBL", "SYTA", "PRGS", "PRCH", "NCNC", "CRNT", "INTU", "LSTR",
    "DUFRY", "RPD", "FUFU", "INVA", "AIRE", "EUSP", "RITM", "NEOG", "BMMJ", "LIDR",
    "SWVL", "CACI", "GPS", "ASIX", "WAB", "TRAW", "FBIN", "VTS", "SOWG", "BOXDQ",
    "GAIN", "GRWG", "SONY", "KELYB", "BE", "GRRR", "RAPP", "TAIT", "V", "BOXL",
    "AHR", "TAKOF", "RWT", "NAVI", "UNCRY", "SOBR", "CREG", "FCX", "CYCN", "PAX",
    "PRTS", "TNON", "SDGR", "HYAC", "KNDI", "GRAF", "FINV", "SMMT", "HYPR", "PW",
    "CF", "BSET", "OBDC", "JCSE", "IAC", "SUHJY", "DSGX", "WU", "BZAMF", "AGNC",
    "DBVT", "ADSK", "AQU", "BW", "ORI", "APAM", "TECH", "BLTE", "IGIC", "WEN",
    "ALTS", "GKOS", "VTMX", "SWKS", "NIO", "ICAD", "HSY", "PRTG", "FMX", "NWG",
    "SIDU", "RJF", "XYLO", "UIS", "JFU", "USEA", "ADN", "CWEN", "DMSL", "VFS",
    "TPST", "FRSX", "CBSH", "FTK", "GBX", "GATO", "XTLB", "DRH", "BUD", "AMC",
    "MDAI", "KIM", "BCOV", "HLLY", "PLTK", "PTLO", "TCRT", "PBA", "ARTL", "ORC",
    "ESYJY", "DKNG", "PAYS", "ITOS", "MRCY", "AMTX", "MCK", "BRDSQ", "TECX", "DRUG",
    "SJ", "BWMX", "REPL", "MAR", "KYMR", "TTEK", "HLVX", "PHXM", "LOGI", "CARG",
    "VRAYQ", "WD", "AQB", "SSIC", "ATMV", "ABVX", "NN", "VZIO", "SBSNY", "VCYT",
    "CDE", "APGE", "CIVI", "MAT", "NMR", "DCGO", "BNCDY", "GIGM", "IVT", "GRPN",
    "MNTS", "ALIT", "SGRP", "MPW", "PLUR", "CBSTF", "HPE", "BRC", "CPPTL", "ICL",
    "BSY", "WB", "LZ", "LSXMA", "BGS", "ORLY", "GGG", "SPRY", "AIXI", "CNBX",
    "OPHC", "KGC", "ACHV", "PTGX", "DTSS", "QNRX", "FCUV", "EVAX", "NTST", "SNCR",
    "INTZ", "MSCI", "NFGC", "GRMN", "CANF", "XBIO", "ARDT", "APM", "AFG", "RICK",
    "PSBD", "FUTU", "RENO", "AEO", "HNVR", "SDRL", "VRAX", "LAUR", "CRESY", "PLXPQ",
    "UHT", "PULM", "ML", "ARMK", "TMUS", "CODI", "EAT", "SEE", "CRBP", "HIW",
    "SLRX", "FOXO", "ASMB", "RGR", "HOFT", "NMM", "XPL", "GFL", "FDSB", "ASAPQ",
    "PKST", "WPM", "TX", "ZBH", "IVAC", "PAYC", "PAVS", "NVAX", "GNFT", "BYDDY",
    "LECO", "CE", "ACU", "CSGP", "REFI", "CAH", "SPKL", "WKSP", "LEGN", "GRTX",
    "AKTS", "RDNT", "SMNNY", "LCTC", "INDO", "EONGY", "CGTX", "DTE", "SEVCF", "TBLA",
    "GLDG", "BBWI", "FND", "AKAM", "GH", "CLPR", "NUE", "K", "MRUS", "ADGM",
    "MPNGY", "AAMC", "HRI", "NRG", "RBTC", "MGTI", "RCRT", "TATT", "ZEVY", "SHLS",
    "AEHL", "MDRR", "NJDCY", "SBFM", "AVIR", "NBRVF", "LAD", "BRN", "QTTOY", "ZONE",
    "ATUSF", "MDLM", "MEDIF", "FANG", "TSLX", "URI", "WMB", "MTTR", "CLFD", "VTAK",
    "CVI", "WDAY", "IP", "CHRW", "TNC", "LYFT", "ARDS", "EGIO", "PETZ", "FCEL",
    "AGFY", "SYY", "CLAR", "ATOS", "KTOS", "JXJT", "KZIA", "HOWL", "CNDT", "TIPT",
    "LKQ", "TEL", "MIST", "CVLT", "AZPN", "PHVS", "XERS", "GMGI", "TRGP", "LUKOY",
    "LND", "BIAF", "IPG", "DRD", "SIM", "FTEL", "IDAI", "AVGO", "CRGX", "WSM",
    "PACB", "SXTC", "SCNI", "NG", "HPQ", "ELAB", "UVV", "PHIO", "COHN", "DUO",
    "NFE", "BRNS", "HYLN", "AIAD", "ARM", "CFRXQ", "GENN", "HOPE", "GSM", "SLRN",
    "UCTT", "WDC", "OPK", "BILL", "SLNO", "CRDO", "RVLV", "MSEX", "DOV", "JRSS",
    "ERJ", "IRBT", "ECO", "MRSN", "AMS", "SSUNF", "PCYO", "LULU", "ARBB", "CMCSA",
    "PEG", "HOUR", "ETON", "MX", "LOT", "SPSC", "CHUC", "UWMC", "SGH", "MYNZ",
    "PBR", "ASLN", "ENVX", "ATLX", "BIOCQ", "ODFL", "BXP", "AMGN", "VJET", "SCVL",
    "SONX", "BMBL", "AVT", "HALMY", "MRK", "ULTA", "RFL", "GIB", "TNFA", "JLL",
    "LQMT", "TRVI", "AAT", "ICCT", "SHOP", "TJX", "AWIN", "SATS", "LQR", "CIFR",
    "ONCT", "CRSR", "BTBD", "ON", "SOTK", "ARVLF", "WCN", "GNRC", "SFTGQ", "RDZN",
    "MMYT", "MTD", "SYBX", "NXGL", "WGS", "CYADY", "ETNB", "SEMR", "EHTH", "YRD",
    "MEOH", "OPRX", "CHSN", "NOTE", "NOV", "ITUB", "YETI", "HYZN", "LPX", "DIST",
    "NANX", "FTNT", "FYBR", "AMCX", "ENIC", "ZH", "KVYO", "SBET", "MATV", "FXLV",
    "VST", "IRDM", "JPPYY", "RECAF", "DOCS", "ADTX", "GTLB", "KPLT", "DASTY", "NSC",
    "FLUT", "CRM", "EVEX", "LFWD", "BCAN", "AXSM", "BSFC", "BTOC", "USEG", "QLYS",
    "PETS", "RIOT", "ASTI", "RC", "ABBNY", "ESEA", "AIRG", "FOUR", "JEWL", "AGS",
    "CTNM", "EP", "UNTC", "RMTI", "BELFB", "GFR", "SBEV", "DBX", "ENLV", "KPTI",
    "JD", "GS", "SLAB", "GWW", "CUBI", "AVTE", "MOH", "NOVS", "EPIX", "PSN",
    "AIR", "KEP", "ALMS", "INTR", "JRJCY", "SMSI", "FIGS", "PTEN", "TLRY", "DIS",
    "EPAM", "PLNT", "DSGT", "HAE", "HZO", "KIDS", "CLRD", "TPVG", "PAG", "DASH",
    "ISPO", "DRI", "HUBC", "OABI", "HSCS", "SMTSF", "KMX", "DLPN", "SFWL", "FIVE",
    "NXST", "VRSN", "QNTM", "SNPS", "AACG", "TFPM", "YELLQ", "LU", "SONO", "MDT",
    "MRIN", "PKG", "DSP", "PPBT", "TNXP", "FRGT", "GHM", "RYTM", "TRVN", "ENR",
    "BNTX", "PRU", "ASPI", "GP", "BTI", "CASI", "JOB", "HPK", "CONNQ", "LEDS",
    "NIPG", "CUTR", "BFRG", "SMLR", "OLLI", "CZR", "DTIL", "EPR", "DSBX", "YIBO",
    "ABVC", "VIVE", "AIV", "CWEGF", "KALRQ", "VCNX", "TWKS", "VERA", "NHTC", "BCH",
    "BIRK", "BCTX", "MDU", "TLPH", "EDR", "CG", "LGCL", "FMST", "GDC", "ONEW",
    "TPB", "PSFE", "BOWN", "AMRN", "YYAI", "TALK", "ARZGY", "SHCO", "MCD", "YOU",
    "ZOM", "SLNH", "OLO", "PIII", "CTRN", "FTCHQ", "TCTM", "VYX", "ENZN", "ATUS",
    "NAUT", "CDTX", "DMEHF", "KLIC", "CUBE", "MTSUY", "ATMC", "FBIO", "RGTI", "WTI",
    "CLRB", "RNECY", "BRT", "VRN", "MIR", "AREN", "DHER", "BGNE", "PSEC", "VRCA",
    "IPSC", "GWRE", "DXYZ", "SPI", "AVGR", "ZLAB", "LKCO", "LB", "GLT", "LRN",
    "NXRT", "SYRE", "EPM", "THRD", "KMI", "NOK", "RUSHA", "MARK", "DNN", "TRI",
    "PFE", "VSME", "ANIP", "CON", "ABIO", "NVCR", "IMAX", "HYREQ", "WEL", "ASML",
    "T", "AKRO", "AHPIQ", "MESA", "AFMD", "CNH", "MTL", "PAM", "BHIL", "EB",
    "FRPT", "WISA", "ACLS", "WSR", "TSM", "EBIXQ", "NCLH", "ARCT", "TBBB", "AMKBY",
    "FOSL", "IGC", "VTR", "AEHR", "CNK", "HOOK", "JBL", "PEV", "PARAA", "MTEK",
    "FBK", "TISI", "STT", "NINOY", "PHUN", "ATHM", "FLUX", "BTG", "RES", "PH",
    "COHU", "ROL", "VYNE", "EXPRQ", "LZB", "AMBP", "GRNQ", "BLKB", "ARCB", "BATRA",
    "PBF", "GPRK", "MERC", "MTSI", "HNRG", "GRAB", "AQN", "XPON", "ZK", "BNGO",
    "KEX", "AWRE", "ADDYY", "NPSNY", "JPM", "ERNA", "CNTY", "HPGLY", "OIBZQ", "PGRE",
    "STKH", "MC", "RHDGF", "HE", "NCL", "BKSY", "PSA", "GLSI", "HUN", "INTG",
    "JFIN", "AVTR", "BRSP", "OPTT", "IQST", "VRE", "UNP", "BGFV", "AU", "WS",
    "SIFY", "XNET", "NILSY", "BIP", "NMTRQ", "ACHL", "FSM", "AIEV", "XHG", "CGAU",
    "KXIN", "VIVK", "EOG", "NMRK", "CLX", "PRQR", "PODC", "STNG", "SDZNY", "AMTD",
    "VFC", "METC", "ARQT", "CHRD", "CNS", "GENI", "NNOMF", "TRU", "SES", "DDL",
    "MDXG", "CRBU", "ECBK", "SYM", "BOLD", "BYD", "CIG", "TCPC", "TWLO", "LMPX",
    "CCLD", "DRCT", "NRIX", "SPPL", "NXPL", "WORX", "PNFP", "MOHOY", "BYSI", "KRUS",
    "AEI", "BRO", "AWX", "ICG", "KD", "BCAB", "DSVSF", "IPWR", "UAA", "ZEPP",
    "MARA", "RRGB", "HOTH", "YXT", "VEOEY", "ALYAF", "PCTY", "KGEI", "ALNY", "EQT",
    "TOST", "SAM", "PTN", "LOBO", "NOVT", "PCRHY", "PMCB", "ADT", "LIVE", "CHH",
    "LHX", "GOOD", "BSX", "ATXG", "SGU", "CABGY", "GPAT", "IMMR", "ATIF", "NWBO",
    "GCT", "PFGC", "MVLA", "GNLX", "MGX", "SGMT", "PSTV", "THCH", "AMD", "DKILY",
    "CSWC", "LTCH", "ITRI", "ARR", "QETA", "CMTG", "CTAS", "CLNN", "CALA", "GGB",
    "OTRK", "MBOT", "ASX", "IFRX", "CERS", "FLO", "ASPS", "GOVX", "APLE", "GHC",
    "LTMCF", "YELP", "NIU", "CVLG", "BTSG", "LXEO", "ALGM", "CUB", "VLO", "GSBD",
    "FRES", "RUM", "SWIM", "CWK", "ECDA", "HST", "MFIC", "NUZE", "FUUFF", "RYN",
    "HTGC", "GIS", "TFSL", "PYXS", "TRT", "COMM", "PBI", "METCB", "CPAY", "XPEL",
    "ZCAR", "CTRI", "NXTP", "LIN", "UAL", "RCAR", "MI", "AITR", "VSTS", "NAAS",
    "TRN", "MDV", "CPB", "TSLA", "CL", "RIG", "AVNW", "GSIW", "PXLW", "EXTR",
    "AJG", "DNOPY", "PSTL", "QBTS", "ALAB", "ZSANQ", "MUX", "SAIC", "WKHS", "KRT",
    "RAIL", "ATGL", "CPNG", "HLXB", "INMD", "TROO", "CRNC", "SNA", "LICN", "ANGH",
    "OMH", "SVC", "AAL", "EPRX", "GPAK", "YEXT", "AMBA", "FCNCA", "RENE", "MANH",
    "ZUO", "TXMD", "SGHC", "KNX", "TGTX", "MRVL", "LZM", "SBLK", "FOXA", "BON",
    "STSS", "HRL", "MAN", "UBS", "CION", "UBX", "LAMR", "FUVV", "OPAD", "ACN",
    "JMHLY", "PCG", "HIPO", "PNRG", "GNW", "MNSO", "GBIO", "DUOL", "EGLXF", "SHIM",
    "SIGA", "LGCB", "GEN", "YY", "BBLG", "MEIP", "FWONK", "TRNS", "ARHS", "COMS",
    "KBH", "ONL", "PEARQ", "GELYY", "FSRNQ", "ANEB", "LEVI", "OSS", "AMPH", "MTUS",
    "SHCR", "LKNCY", "RPTX", "BRQSF", "CARV", "INTS", "TEM", "OSPN", "RKT", "YGMZ",
    "BLIN", "PNGAY", "EVLO", "TROW", "RVTY", "GTHX", "EZGO", "XISHY", "ACXP", "BWEN",
    "PTCT", "MGOL", "CYDVF", "RLMD", "GOEV", "FDS", "CEVA", "ELV", "KOSS", "PEBO",
    "SLSR", "RYI", "CTRM", "GAUZ", "RPHM", "COST", "IXHL", "VEEE", "NEXI", "KGS",
    "VNOM", "BALL", "RVPH", "CZOOF", "PCH", "VRNA", "IPAR", "ACI", "PAYX", "ASB",
    "WTO", "TRUE", "RMCF", "MIELY", "SLS", "YYGH", "FRO", "TRX", "JENGQ", "OXY",
    "CUK", "EOLS", "HAS", "BEPC", "FIS", "ETD", "LNTH", "GOOS", "AXLA", "NBIX",
    "VLD", "SVII", "LMT", "ATHE", "CME", "MAS", "HLF", "TYGO", "SIEB", "FDP",
    "BATRK", "SCI", "MTB", "NEXT", "CCOEY", "CAKE", "CIBEY", "HDB", "CLH", "CELC",
    "ATNF", "RAY", "AA", "CRKN", "APD", "EHGO", "SRCL", "KODK", "WRNT", "CYRNQ",
    "PSNY", "IRM", "RCAT", "STOK", "CPOP", "KAVL", "XEL", "WNW", "CW", "MMAT",
    "TRMB", "GDRX", "BKE", "WMG", "PLG", "LNG", "IRS", "CLF", "YCBD", "MTNB",
    "SHEL", "CRGY", "CANG", "CHCT", "IPA", "PXS", "BEP", "LLYVA", "RBLX", "OCTO",
    "BIIB", "TSCO", "OB", "ACOPY", "CPAC", "SLE", "WNDW", "PKKFF", "FNKO", "LMFA",
    "INN", "KRC", "NCNA", "GSAT", "LNN", "CDAKQ", "GLPI", "TTCFQ", "NOGNQ", "YUM",
    "TOP", "BIPC", "JBHT", "KYTX", "SOS", "KURA", "AGCO", "HTOO", "AOS", "SRE",
    "COLD", "SHOO", "FLL", "RY", "SOHO", "TRIRF", "ACST", "JTAI", "UTSI", "FREY",
    "ESS", "NVX", "CMCT", "IDEX", "SWBI", "RUBY", "BLMZ", "OSK", "AXDX", "CBL",
    "SN", "CKHUY", "AMG", "SDOT", "CREX", "BOLT", "PFG", "KSS", "CLRO", "BORR",
    "BDL", "FIX", "CHKP", "OXLC", "SAR", "EKSO", "EAF", "AZTR", "SHAK", "MORN",
    "NVTS", "LCID", "CNXC", "CIO", "TWST", "POET", "GLATF", "POWW", "BUJA", "BP",
    "ABG", "HYFM", "E", "TCS", "MHVYF", "EVI", "UCL", "DTST", "SRGHY", "IZM",
    "NTDOY", "PRI", "MYPS", "NWTN", "NAOV", "SUZ", "BDRL", "RILY", "IQV", "CRVO",
    "BCRX", "JVSA", "BTTX", "TBPH", "MTZ", "SU", "EQR", "IDN", "NHI", "LASE",
    "MAXN", "LPTH", "CATX", "DYNT", "VSEE", "SNAX", "OMI", "MGRX", "ONMD", "SSSS",
    "CTLT", "SUNWQ", "FANUY", "IBAC", "EDSA", "LOCL", "ZIMV", "COGT", "KHC", "CNGL",
    "ACTG", "FHTX", "RERE", "BTM", "WTS", "DX", "BCDA", "KNF", "ATAI", "IVVD",
    "SXC", "GLBE", "ORIC", "QMCO", "APVO", "ZCMD", "SKX", "TKC", "LITB", "DYCQ",
    "BPMC", "HESM", "FIHL", "CDNA", "MXCT", "AUUD", "UAMY", "CMBM", "DTCK", "ENVB",
    "SQM", "NDAQ", "ATER", "YGFGF", "SVV", "VCSA", "GT", "AIRI", "FAMI", "VITL",
    "EVSBY", "INZY", "LAB", "CVKD", "PRFT", "KKR", "FSUGY", "BKCC", "R", "COKE",
    "HCC", "ACRE", "ADPT", "PROF", "ANTX", "BOH", "DPMLF", "CHALF", "CING", "MGPI",
    "BFAM", "BBDO", "WCRS", "APDN", "PROP", "SUUN", "IT", "AVBP", "BAYA", "RNA",
    "TGT", "SEOAY", "XOMA", "FGEN", "FLWS", "ENPH", "AMPX", "SCPS", "CWH", "TELL",
    "CABO", "EFX", "FSK", "CTRE", "SIG", "ZTS", "FRPH", "CYRX", "YI", "LEXX",
    "AI", "RNXT", "ALLR", "ARGX", "TENB", "GLASF", "HSDT", "GORO", "ROIC", "RGLD",
    "BEKE", "LX", "BCAT", "NCMI", "AVDL", "NNOX", "DOUG", "RAYA", "HSHP", "XIN",
    "NLY", "RCON", "LEN", "XOM", "TELO", "JFBR", "ELA", "ANL", "JUNE", "FDX",
    "ORKT", "HII", "PRTA", "KKOYY", "UNCY", "IRIX", "NUVL", "NMG", "TRMLF", "APPN",
    "DLR", "RMR", "UTHR", "VIV", "AIOT", "STI", "VZ", "VET", "BMRN", "PMTS",
    "COCO", "TD", "NOTV", "CHWY", "VIPS", "APP", "BBW", "RLAY", "LSH", "BNL",
    "CSCO", "TAP", "BRTX", "EEMMF", "BEAM", "SEVN", "EIGRQ", "PRGO", "ABBV", "YTEN",
    "ADTN", "CSL", "SOBKY", "BDX", "HEPS", "TRNFQ", "UTI", "QH", "NFSCF", "NLOP",
    "DD", "PFLT", "QMCI", "OIGBQ", "BL", "CSTL", "APH", "SYPR", "DOMA", "AULT",
    "BCE", "PAL", "GLAD", "MRX", "MULN", "ICE", "DHI", "BLCO", "SKYH", "LOOP",
    "CBGPY", "ALXO", "KVAC", "LOW", "NTCOY", "TPX", "NVR", "KRKR", "CAT", "OAKU",
    "REPYY", "SPWH", "CNTM", "CAPR", "SILV", "RVNC", "TIOG", "DSS", "DOX", "AGBA",
    "STEM", "UNH", "XPOF", "BKYI", "NVDA", "USB", "SECOY", "BROS", "EVR", "AFCG",
    "VUZI", "CYN", "WATT", "GNS", "IPNFF", "MVST", "FSUN", "FORD", "MKSI", "BTE",
    "MCOM", "CALM", "STZ", "CENX", "VTRS", "UFABQ", "HG", "AFIB", "OGI", "XGN",
    "CARA", "PKX", "BFAC", "CDNS", "MTBLY", "MNY", "OTLK", "CSV", "SABS", "AMN",
    "NKGN", "MAC", "ONCSQ", "NKRKY", "ASPN", "TBI", "NUS", "RH", "NWN", "DATS",
    "IVR", "COF", "MKFG", "TMO", "ESAB", "GRIN", "NEP", "HD", "ARES", "LGVN",
    "CVNA", "RSI", "RNAC", "TRNR", "ACCD", "EL", "OLPX", "NLSP", "INVZ", "ARCC",
    "ELPC", "EXAS", "VINO", "NPK", "JNCE", "RPID", "OXSQ", "HIMX", "KDP", "AEIS",
    "SWRAY", "BBIO", "PSNL", "SELF", "CKPT", "DB", "TYL", "WING", "HSIC", "EZFL",
    "GVH", "CUEN", "SKYX", "ECVT", "MSI", "SUI", "NXDT", "YORW", "STVN", "BKH",
    "FLNC", "ORGO", "SWX", "LLESY", "WBEVQ", "HSPO", "BFXXQ", "TDUP", "ZD", "SPGC",
    "TGI", "ACGN", "FRFHF", "CTSO", "AGRI", "MBC", "PNRLF", "BZFD", "TAL", "OCX",
    "UPWK", "JOUT", "PPSI", "RZLT", "BOOT", "VERU", "OXBR", "AURA", "MJARF", "ED",
    "DGLY", "PYRGF", "BRLT", "HEAR", "WHLM", "TLGTQ", "FPBC", "QSG", "PHASQ", "ILAG",
    "STRL", "JAMF", "TKO", "FSLR", "TOPPY", "FLS", "BRCC", "MED", "LOCO", "BIVI",
    "MVIS", "DMKPQ", "MLM", "ADSE", "OUT", "CURI", "EA", "MO", "VTNR", "PSX",
    "NMTC", "ACRDF", "HOVR", "ALLY", "RTX", "NUVO", "EGP", "CDP", "COLB", "OLN",
    "AIRS", "OMF", "NMRD", "ATLC", "AMCR", "BASE", "EBC", "JKS", "SGRY", "EM",
    "JACK", "OPNT", "BYRN", "CGC", "DHT", "NLCP", "KOS", "GHRS", "INVH", "SOI",
    "PKE", "EPRT", "AAME", "MOMO", "OSUR", "CENN", "SCPH", "RIVN", "EVLV", "PK",
    "PCOR", "UCAR", "SPR", "ZIM", "KEQU", "TT", "RYAAY", "EXPI", "ALTR", "BURL",
    "ISPC", "PRMW", "SLN", "IPI", "SSRM", "CPRX", "PANL", "UOVEY", "SST", "NEGG",
    "OKLO", "IMNM", "WSO", "FER", "TDS", "AIG", "HDL", "WYY", "DIOD", "VTLE",
    "CORZ", "HUYA", "WAL", "TRIP", "GOLF", "BTBT", "IMBIQ", "ABAT", "AKR", "TOVX",
    "HXGBY", "NKE", "PEP", "OBE", "TEN", "SLDB", "PAAS", "MRAAY", "RAPT", "PRTK",
    "HSTO", "BK", "ADM", "ONFO", "ADC", "GOSS", "BSBR", "MOD", "JAZZ", "TK",
    "CHK", "DCFCQ", "INOD", "EMR", "MFH", "CISO", "DRTS", "FE", "STER", "ETRGF",
    "CLNE", "CECO", "HUM", "APLT", "TRVG", "CHTR", "AMR", "MQBKY", "GASS", "RKDA",
    "CXDC", "ULY", "HAFN", "HKD", "LOAN", "UONEK", "DCBO", "NWL", "CNTTQ", "LVMUY",
    "DCO", "CWCO", "MDLZ", "TORO", "RMBS", "PDM", "DFLI", "MCRB", "RNVA", "NYMT",
    "NSA", "SISI", "RBBN", "ACHC", "SYK", "NCNO", "LSXMK", "PMT", "SVBL", "SQSP",
    "CYTK", "ZDGE", "AXP", "RDDT", "BTU", "VSTO", "HCA", "HNRA", "ACRS", "VSTA",
    "SKYT", "GENE", "GMVDF", "RL", "PSHG", "RGEN", "RETO", "SRAD", "LRCX", "XXII",
    "ROOT", "JMIA", "YITYY", "BNS", "LAZR", "LDI", "BDNNY", "LIFW", "NIVF", "AUR",
    "UBSFY", "SPWRQ", "SBUX", "CPBI", "PAGP", "ROST", "CLVT", "CBAT", "BMTX", "EXEL",
    "HL", "DNMR", "POOL", "AON", "ITRN", "NWBI", "SLQT", "BENF", "SEED", "EGO",
    "AMIX", "MDB", "CEG", "PRYMY", "SBSW", "BAX", "OIS", "BANT", "PAYO", "TMPOQ",
    "NRSN", "TECK", "GPK", "INNV", "CIM", "CEIN", "CALX", "FLR", "ESLT", "LCTX",
    "BOTJ", "RANI", "WKC", "CLBT", "SRTS", "CETX", "ALVO", "KRRO", "OKE", "PETV",
    "PIK", "LTM", "VRNT", "DNOW", "MEEC", "IMVT", "DE", "HP", "HBM", "WBD",
    "REI", "CBDBY", "GTII", "AEM", "WBA", "ACAD", "WLK", "NVVE", "FBRX", "YQ",
    "TGLS", "LYT", "BAC", "INVE", "AGLXY", "CTSH", "SIVBQ", "ANF", "LYSDY", "AVXL",
    "BBY", "BSVN", "APTO", "APA", "VNT", "BJ", "DBD", "HUIZ", "HMC", "CRL",
    "CYTO", "CURLF", "MTCH", "LRMR", "PRE", "ME", "DNUT", "TSAT", "STNE", "MTLS",
    "CPHI", "VERI", "MTN", "THFF", "TOPS", "SNPX", "SYRS", "ARCO", "LIFZF", "TTWO",
    "NOVA", "VRNS", "ACIU", "IRON", "DRRX", "EXLS", "BTCS", "PSTH", "PUMP", "OCEA",
    "ARBK", "WM", "AIHS", "DELKY", "INDV", "GTY", "CLPBY", "SIRI", "BNSOF", "KRMD",
    "CMCL", "DAO", "VVV", "MPWR", "ARTV", "DESP", "MNST", "ECL", "IR", "ITCI",
    "SDVKY", "PLYM", "XSNX", "CLPT", "PSO", "ANPDY", "IOGPQ", "EIX", "MRTI", "ROKU",
    "QSR", "MIRM", "PEGA", "WTRG", "ORGN", "SITM", "AZTA", "AISP", "SVM", "DSY",
    "BMA", "MWA", "FRCB", "MBLY", "PLL", "DEO", "QGEN", "NTB", "WBTN", "WLDS",
    "MGAM", "ENTO", "INFA", "BDTX", "JNJ", "SOND", "PINE", "HNGKY", "VWESQ", "KVUE",
    "MTC", "DRMA", "ABNB", "ELVN", "SFM", "INO", "HASI", "BX", "BTAI", "ADBE",
    "DTEAF", "BFRI", "SCKT", "IVZ", "MYRG", "SMWB", "STLA", "MLP", "PTON", "OC",
    "CXDO", "CI", "ARMP", "BHAT", "OTSKY", "GOOG", "LTRY", "EYEN", "CMC", "GSUN",
    "IMMX", "VERB", "SCLX", "NGKSY", "SCHW", "ATNM", "NAK", "SPXC", "SOLV", "BLND",
    "TSSI", "KPRX", "ALF", "ARDX", "SWIN", "BCEL", "IAUX", "STR", "RNAZ", "CP",
    "POLA", "CCAP", "NNXPF", "DSHK", "OWL", "GEV", "BIO", "FIZZ", "ANSS", "JG",
    "SONN", "MFI", "UPBD", "RDHL", "CRK", "EVER", "TSEM", "DXCM", "CLEU", "HON",
    "BTCM", "MLYS", "MIMOQ", "EVH", "CLCO", "DXC", "SE", "APYX", "HALO", "WBUY",
    "XPRO", "CDXS", "CCOI",]

def load_existing(path="tickers_cleaned.txt"):
    if not os.path.exists(path):
        print("ERROR: tickers_cleaned.txt not found. Run refresh_tickers.py first.")
        exit(1)
    return set(t.strip() for t in open(path) if t.strip())

def check_batch(batch):
    passing = []
    try:
        data = yf.download(batch, period="15d", progress=False, auto_adjust=True)
        if data.empty:
            return passing
        closes  = data["Close"].iloc[-1]
        volumes = data["Volume"].tail(10).mean()
        for t in batch:
            try:
                price = float(closes[t]) if t in closes.index else None
                vol   = float(volumes[t]) if t in volumes.index else None
                if price is None or pd.isna(price):
                    continue
                if price < PRICE_MIN or price > PRICE_MAX:
                    continue
                if vol is None or pd.isna(vol) or vol < VOL_MIN:
                    continue
                passing.append(t)
            except Exception:
                continue
    except Exception:
        pass
    return passing

existing = load_existing()
slots_needed = TARGET - len(existing)

print(f"Current clean tickers: {len(existing)}")
print(f"Target: {TARGET}")
print(f"Slots to fill: {slots_needed}")

if slots_needed <= 0:
    print("Already at target. Writing tickers_final.txt.")
    with open("tickers_final.txt", "w") as f:
        for t in sorted(existing):
            f.write(t + "\n")
    exit(0)

# Filter candidates before checking prices
raw_candidates = T212_US_TICKERS
pre_filtered = []
skipped_etf = 0
skipped_pattern = 0
for t in raw_candidates:
    t = t.strip()
    if not t:
        continue
    if t in EXCLUDED_TICKERS:
        skipped_etf += 1
        continue
    if is_excluded_by_pattern(t):
        skipped_pattern += 1
        continue
    pre_filtered.append(t)

# Remove already-in-list tickers
candidates = []
seen = set(existing)
for t in pre_filtered:
    if t not in seen:
        candidates.append(t)
        seen.add(t)

print(f"\nPre-filtering:")
print(f"  Excluded known ETFs/funds: {skipped_etf}")
print(f"  Excluded by ticker pattern: {skipped_pattern}")
print(f"  Candidates to price-check: {len(candidates)}")
print(f"  Volume minimum: {VOL_MIN:,} shares/day")
print(f"\nChecking prices... this takes 5-10 minutes.\n")

new_tickers = []
checked = 0

for i in range(0, len(candidates), BATCH_SIZE):
    if len(new_tickers) >= slots_needed:
        break
    batch = candidates[i:i+BATCH_SIZE]
    passing = check_batch(batch)
    new_tickers.extend(passing)
    checked += len(batch)
    still_needed = slots_needed - len(new_tickers)
    print(f"  Checked {checked} | Found {len(new_tickers)}/{slots_needed} | Need {still_needed} more")
    time.sleep(0.5)

new_tickers = new_tickers[:slots_needed]
final = sorted(existing | set(new_tickers))

with open("tickers_final.txt", "w") as f:
    for t in final:
        f.write(t + "\n")

print("=" * 50)
print("Done!")
print(f"  Kept from cleaned list:     {len(existing)}")
print(f"  New tickers added:          {len(new_tickers)}")
print(f"  Total in tickers_final.txt: {len(final)}")
print("\nNext steps:")
print("  1. In your repo folder, rename tickers.txt to tickers_backup.txt")
print("  2. Rename tickers_final.txt to tickers.txt")
print("  3. Open GitHub Desktop, commit and push")