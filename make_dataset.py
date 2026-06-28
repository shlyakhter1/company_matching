#!/usr/bin/env python3
"""
make_dataset.py — build a labeled company-name matching benchmark.

~120 ground-truth clusters of real companies, each with several name variants
(suffix/abbrev/acronym/typo/translit/former-name/ticker/...), plus deliberately
hard "confusable" families (United Airlines vs United Parcel, Merck & Co vs
Merck KGaA, GM vs GE vs General Mills, HP Inc vs HPE, ...).

Outputs (to --outdir, default ./data):
  company_records.csv : one row per name variant, with ground-truth cluster_id
  company_pairs.csv   : labeled match / non-match pairs (positive/hard/easy)

Usage:
  python make_dataset.py                       # full ~120-cluster set
  python make_dataset.py --max-clusters 40     # subset for a quick run
  python make_dataset.py --seed 7 --outdir data
"""
import csv, itertools, random, re, argparse, os

# ---------------------------------------------------------------------------
# cluster_id : (canonical, country, [(variant, variation_type), ...])
# variation_type vocab:
#   base suffix abbrev acronym ampersand the_prefix spacing_punct typo
#   translit former_name ticker casing
# ---------------------------------------------------------------------------
CLUSTERS = {
 # ---- Tech / software / internet ----
 "apple": ("Apple Inc.","US",[("Apple Inc.","base"),("Apple Computer, Inc.","former_name"),("Apple","suffix"),("AAPL","ticker"),("Aple Inc","typo")]),
 "microsoft": ("Microsoft Corporation","US",[("Microsoft Corporation","base"),("Microsoft Corp.","abbrev"),("MSFT","ticker"),("Microsft","typo")]),
 "google": ("Google LLC","US",[("Google LLC","base"),("Google Inc.","former_name"),("Google","suffix")]),
 "alphabet": ("Alphabet Inc.","US",[("Alphabet Inc.","base"),("Alphabet","suffix"),("GOOGL","ticker")]),
 "meta": ("Meta Platforms, Inc.","US",[("Meta Platforms, Inc.","base"),("Facebook, Inc.","former_name"),("Meta","suffix"),("META","ticker")]),
 "amazon": ("Amazon.com, Inc.","US",[("Amazon.com, Inc.","base"),("Amazon","suffix"),("Amazon.com","spacing_punct"),("AMZN","ticker")]),
 "ibm": ("International Business Machines Corporation","US",[("International Business Machines Corporation","base"),("IBM","acronym"),("I.B.M.","acronym"),("IBM Corp.","abbrev"),("Internationl Business Machines","typo")]),
 "oracle": ("Oracle Corporation","US",[("Oracle Corporation","base"),("Oracle Corp.","abbrev"),("Oracle","suffix")]),
 "sap": ("SAP SE","DE",[("SAP SE","base"),("SAP AG","former_name"),("SAP","suffix")]),
 "salesforce": ("Salesforce, Inc.","US",[("Salesforce, Inc.","base"),("Salesforce.com, Inc.","former_name"),("Salesforce","suffix"),("CRM","ticker")]),
 "adobe": ("Adobe Inc.","US",[("Adobe Inc.","base"),("Adobe Systems Incorporated","former_name"),("Adobe","suffix")]),
 "servicenow": ("ServiceNow, Inc.","US",[("ServiceNow, Inc.","base"),("ServiceNow","suffix"),("Service Now","spacing_punct")]),
 "netflix": ("Netflix, Inc.","US",[("Netflix, Inc.","base"),("Netflix","suffix"),("NFLX","ticker")]),
 "uber": ("Uber Technologies, Inc.","US",[("Uber Technologies, Inc.","base"),("Uber","suffix"),("UBER","ticker")]),
 "airbnb": ("Airbnb, Inc.","US",[("Airbnb, Inc.","base"),("Airbnb","suffix"),("Air BnB","spacing_punct")]),
 "paypal": ("PayPal Holdings, Inc.","US",[("PayPal Holdings, Inc.","base"),("PayPal","suffix"),("Pay Pal","spacing_punct")]),
 "block": ("Block, Inc.","US",[("Block, Inc.","base"),("Square, Inc.","former_name"),("Block","suffix")]),
 "snowflake": ("Snowflake Inc.","US",[("Snowflake Inc.","base"),("Snowflake","suffix"),("SNOW","ticker")]),
 "palantir": ("Palantir Technologies Inc.","US",[("Palantir Technologies Inc.","base"),("Palantir","suffix"),("PLTR","ticker")]),
 "spotify": ("Spotify Technology S.A.","LU",[("Spotify Technology S.A.","base"),("Spotify","suffix"),("Spotify Technology","abbrev")]),
 "dell": ("Dell Technologies Inc.","US",[("Dell Technologies Inc.","base"),("Dell Inc.","former_name"),("Dell","suffix")]),
 "lenovo": ("Lenovo Group Limited","CN",[("Lenovo Group Limited","base"),("Lenovo","suffix"),("联想集团","translit")]),
 # ---- Semiconductors ----
 "nvidia": ("NVIDIA Corporation","US",[("NVIDIA Corporation","base"),("Nvidia","casing"),("NVDA","ticker"),("Nvidia Corp","abbrev")]),
 "amd": ("Advanced Micro Devices, Inc.","US",[("Advanced Micro Devices, Inc.","base"),("AMD","acronym"),("Advanced Micro Devices","suffix")]),
 "intel": ("Intel Corporation","US",[("Intel Corporation","base"),("Intel Corp","abbrev"),("INTC","ticker")]),
 "qualcomm": ("QUALCOMM Incorporated","US",[("QUALCOMM Incorporated","base"),("Qualcomm","casing"),("QCOM","ticker")]),
 "broadcom": ("Broadcom Inc.","US",[("Broadcom Inc.","base"),("Broadcom","suffix"),("AVGO","ticker")]),
 "ti": ("Texas Instruments Incorporated","US",[("Texas Instruments Incorporated","base"),("Texas Instruments","suffix"),("TI","acronym")]),
 "micron": ("Micron Technology, Inc.","US",[("Micron Technology, Inc.","base"),("Micron","suffix"),("MU","ticker")]),
 "tsmc": ("Taiwan Semiconductor Manufacturing Company Limited","TW",[("Taiwan Semiconductor Manufacturing Company Limited","base"),("TSMC","acronym"),("Taiwan Semiconductor","abbrev"),("TSM","ticker")]),
 "asml": ("ASML Holding N.V.","NL",[("ASML Holding N.V.","base"),("ASML","acronym"),("ASML Holding","suffix")]),
 # ---- Pharma / health (incl. the Merck trap) ----
 "merck_us": ("Merck & Co., Inc.","US",[("Merck & Co., Inc.","base"),("Merck and Co","ampersand"),("Merck Sharp & Dohme","former_name"),("MSD","acronym")]),
 "merck_de": ("Merck KGaA","DE",[("Merck KGaA","base"),("Merck Group","suffix"),("E. Merck","former_name")]),
 "pfizer": ("Pfizer Inc.","US",[("Pfizer Inc.","base"),("Pfizer","suffix"),("PFE","ticker")]),
 "novartis": ("Novartis AG","CH",[("Novartis AG","base"),("Novartis","suffix"),("Novartis International AG","abbrev")]),
 "roche": ("Roche Holding AG","CH",[("Roche Holding AG","base"),("F. Hoffmann-La Roche","former_name"),("Roche","suffix")]),
 "astrazeneca": ("AstraZeneca PLC","GB",[("AstraZeneca PLC","base"),("AstraZeneca","suffix"),("Astra Zeneca","spacing_punct"),("AZN","ticker")]),
 "gsk": ("GSK plc","GB",[("GSK plc","base"),("GlaxoSmithKline plc","former_name"),("GlaxoSmithKline","suffix"),("GSK","acronym")]),
 "sanofi": ("Sanofi S.A.","FR",[("Sanofi S.A.","base"),("Sanofi","suffix"),("Sanofi-Aventis","former_name")]),
 "bayer": ("Bayer AG","DE",[("Bayer AG","base"),("Bayer","suffix"),("Bayer Aktiengesellschaft","suffix")]),
 "lilly": ("Eli Lilly and Company","US",[("Eli Lilly and Company","base"),("Eli Lilly","suffix"),("Eli Lilly & Co","ampersand"),("LLY","ticker")]),
 "abbvie": ("AbbVie Inc.","US",[("AbbVie Inc.","base"),("AbbVie","suffix"),("Abbvie","casing")]),
 "bms": ("Bristol-Myers Squibb Company","US",[("Bristol-Myers Squibb Company","base"),("Bristol Myers Squibb","spacing_punct"),("BMS","acronym")]),
 "moderna": ("Moderna, Inc.","US",[("Moderna, Inc.","base"),("Moderna","suffix"),("ModernaTX","former_name")]),
 "amgen": ("Amgen Inc.","US",[("Amgen Inc.","base"),("Amgen","suffix"),("AMGN","ticker")]),
 "jnj": ("Johnson & Johnson","US",[("Johnson & Johnson","base"),("Johnson and Johnson","ampersand"),("J&J","acronym")]),
 # ---- Auto ----
 "toyota": ("Toyota Motor Corporation","JP",[("Toyota Motor Corporation","base"),("Toyota Motor Corp","abbrev"),("Toyota","suffix"),("トヨタ自動車","translit")]),
 "vw": ("Volkswagen AG","DE",[("Volkswagen AG","base"),("Volkswagen Aktiengesellschaft","suffix"),("VW","acronym"),("Volkswagon","typo")]),
 "bmw": ("Bayerische Motoren Werke AG","DE",[("Bayerische Motoren Werke AG","base"),("BMW AG","acronym"),("BMW","acronym")]),
 "mercedes": ("Mercedes-Benz Group AG","DE",[("Mercedes-Benz Group AG","base"),("Daimler AG","former_name"),("Mercedes Benz","spacing_punct"),("Mercedes-Benz","suffix")]),
 "ford": ("Ford Motor Company","US",[("Ford Motor Company","base"),("Ford Motor Co.","abbrev"),("Ford","suffix")]),
 "gm": ("General Motors Company","US",[("General Motors Company","base"),("General Motors","suffix"),("GM","acronym"),("Genral Motors","typo")]),
 "honda": ("Honda Motor Co., Ltd.","JP",[("Honda Motor Co., Ltd.","base"),("Honda","suffix"),("Honda Motor","abbrev"),("本田技研工業","translit")]),
 "stellantis": ("Stellantis N.V.","NL",[("Stellantis N.V.","base"),("Fiat Chrysler Automobiles","former_name"),("Stellantis","suffix")]),
 "nissan": ("Nissan Motor Co., Ltd.","JP",[("Nissan Motor Co., Ltd.","base"),("Nissan","suffix"),("Nissan Motor","abbrev")]),
 "hyundai": ("Hyundai Motor Company","KR",[("Hyundai Motor Company","base"),("Hyundai","suffix"),("현대자동차","translit")]),
 "tesla": ("Tesla, Inc.","US",[("Tesla, Inc.","base"),("Tesla Motors, Inc.","former_name"),("Tesla","suffix"),("TSLA","ticker")]),
 "porsche": ("Dr. Ing. h.c. F. Porsche AG","DE",[("Dr. Ing. h.c. F. Porsche AG","base"),("Porsche AG","abbrev"),("Porsche","suffix")]),
 "renault": ("Renault S.A.","FR",[("Renault S.A.","base"),("Renault","suffix"),("Groupe Renault","suffix")]),
 # ---- Energy ----
 "exxon": ("Exxon Mobil Corporation","US",[("Exxon Mobil Corporation","base"),("ExxonMobil","spacing_punct"),("Exxon","suffix"),("XOM","ticker")]),
 "shell": ("Shell plc","GB",[("Shell plc","base"),("Royal Dutch Shell","former_name"),("Shell","suffix")]),
 "chevron": ("Chevron Corporation","US",[("Chevron Corporation","base"),("Chevron","suffix"),("CVX","ticker")]),
 "bp": ("BP p.l.c.","GB",[("BP p.l.c.","base"),("British Petroleum","former_name"),("BP","acronym")]),
 "total": ("TotalEnergies SE","FR",[("TotalEnergies SE","base"),("Total S.A.","former_name"),("TotalEnergies","suffix"),("Total","abbrev")]),
 "conoco": ("ConocoPhillips","US",[("ConocoPhillips","base"),("Conoco Phillips","spacing_punct"),("COP","ticker")]),
 "aramco": ("Saudi Arabian Oil Company","SA",[("Saudi Arabian Oil Company","base"),("Saudi Aramco","suffix"),("Aramco","abbrev")]),
 "gazprom": ("PJSC Gazprom","RU",[("PJSC Gazprom","base"),("Gazprom","suffix"),("Газпром","translit"),("OAO Gazprom","former_name")]),
 "equinor": ("Equinor ASA","NO",[("Equinor ASA","base"),("Statoil ASA","former_name"),("Equinor","suffix")]),
 "petrobras": ("Petróleo Brasileiro S.A.","BR",[("Petróleo Brasileiro S.A.","base"),("Petrobras","suffix"),("Petroleo Brasileiro","translit")]),
 # ---- Finance / banks / payments ----
 "jpmorgan": ("JPMorgan Chase & Co.","US",[("JPMorgan Chase & Co.","base"),("JP Morgan","abbrev"),("J.P. Morgan","spacing_punct"),("JPMorgan Chase","suffix")]),
 "goldman": ("The Goldman Sachs Group, Inc.","US",[("The Goldman Sachs Group, Inc.","base"),("Goldman Sachs","suffix"),("GS","ticker")]),
 "morganstanley": ("Morgan Stanley","US",[("Morgan Stanley","base"),("Morgan Stanley & Co.","suffix"),("MS","ticker")]),
 "bofa": ("Bank of America Corporation","US",[("Bank of America Corporation","base"),("Bank of America","suffix"),("BofA","abbrev"),("BAC","ticker")]),
 "citi": ("Citigroup Inc.","US",[("Citigroup Inc.","base"),("Citigroup","suffix"),("Citi","abbrev"),("C","ticker")]),
 "wells": ("Wells Fargo & Company","US",[("Wells Fargo & Company","base"),("Wells Fargo and Company","ampersand"),("Wells Fargo","suffix"),("WFC","ticker")]),
 "hsbc": ("HSBC Holdings plc","GB",[("HSBC Holdings plc","base"),("HSBC","acronym"),("Hongkong and Shanghai Banking Corporation","former_name")]),
 "barclays": ("Barclays PLC","GB",[("Barclays PLC","base"),("Barclays","suffix"),("Barclays Bank","suffix")]),
 "bnp": ("BNP Paribas S.A.","FR",[("BNP Paribas S.A.","base"),("BNP Paribas","suffix"),("BNP","abbrev")]),
 "deutschebank": ("Deutsche Bank AG","DE",[("Deutsche Bank AG","base"),("Deutsche Bank","suffix"),("DB","acronym")]),
 "santander": ("Banco Santander, S.A.","ES",[("Banco Santander, S.A.","base"),("Santander","abbrev"),("Banco Santander","suffix")]),
 "ubs": ("UBS Group AG","CH",[("UBS Group AG","base"),("UBS","acronym"),("Union Bank of Switzerland","former_name")]),
 "visa": ("Visa Inc.","US",[("Visa Inc.","base"),("Visa","suffix"),("V","ticker")]),
 "mastercard": ("Mastercard Incorporated","US",[("Mastercard Incorporated","base"),("MasterCard","casing"),("Mastercard","suffix"),("MA","ticker")]),
 "amex": ("American Express Company","US",[("American Express Company","base"),("American Express","suffix"),("Amex","abbrev"),("AXP","ticker")]),
 "berkshire": ("Berkshire Hathaway Inc.","US",[("Berkshire Hathaway Inc.","base"),("Berkshire Hathaway","suffix"),("Berkshire","abbrev"),("BRK.A","ticker")]),
 "blackrock": ("BlackRock, Inc.","US",[("BlackRock, Inc.","base"),("BlackRock","suffix"),("Black Rock","spacing_punct"),("BLK","ticker")]),
 # ---- Retail / consumer / food & bev ----
 "walmart": ("Walmart Inc.","US",[("Walmart Inc.","base"),("Wal-Mart Stores, Inc.","former_name"),("Wal Mart","spacing_punct"),("WMT","ticker")]),
 "costco": ("Costco Wholesale Corporation","US",[("Costco Wholesale Corporation","base"),("Costco","suffix"),("COST","ticker")]),
 "target": ("Target Corporation","US",[("Target Corporation","base"),("Target","suffix"),("TGT","ticker")]),
 "homedepot": ("The Home Depot, Inc.","US",[("The Home Depot, Inc.","base"),("Home Depot","the_prefix"),("The Home Depot","suffix")]),
 "lowes": ("Lowe's Companies, Inc.","US",[("Lowe's Companies, Inc.","base"),("Lowes","spacing_punct"),("Lowe's","suffix")]),
 "alibaba": ("Alibaba Group Holding Limited","CN",[("Alibaba Group Holding Limited","base"),("Alibaba Group","suffix"),("Alibaba","abbrev"),("BABA","ticker")]),
 "nike": ("NIKE, Inc.","US",[("NIKE, Inc.","base"),("Nike","casing"),("Nike Inc","suffix"),("NKE","ticker")]),
 "adidas": ("adidas AG","DE",[("adidas AG","base"),("Adidas","casing"),("adidas","suffix")]),
 "mcdonalds": ("McDonald's Corporation","US",[("McDonald's Corporation","base"),("McDonalds","spacing_punct"),("McDonald's","suffix"),("MCD","ticker")]),
 "starbucks": ("Starbucks Corporation","US",[("Starbucks Corporation","base"),("Starbucks","suffix"),("SBUX","ticker")]),
 "cocacola": ("The Coca-Cola Company","US",[("The Coca-Cola Company","base"),("Coca-Cola Co","suffix"),("Coca Cola","spacing_punct"),("KO","ticker")]),
 "pepsico": ("PepsiCo, Inc.","US",[("PepsiCo, Inc.","base"),("PepsiCo","suffix"),("Pepsi","abbrev"),("PEP","ticker")]),
 "unilever": ("Unilever PLC","GB",[("Unilever PLC","base"),("Unilever","suffix"),("Unilever N.V.","former_name")]),
 "nestle": ("Nestlé S.A.","CH",[("Nestlé S.A.","base"),("Nestle SA","translit"),("Nestle","suffix"),("Société des Produits Nestlé","former_name")]),
 "pg": ("The Procter & Gamble Company","US",[("The Procter & Gamble Company","base"),("Procter & Gamble Co.","suffix"),("Procter and Gamble","ampersand"),("P&G","acronym"),("Proctor & Gamble","typo")]),
 "loreal": ("L'Oréal S.A.","FR",[("L'Oréal S.A.","base"),("L'Oreal","translit"),("LOreal","spacing_punct"),("L'Oréal","suffix")]),
 "inditex": ("Industria de Diseño Textil, S.A.","ES",[("Industria de Diseño Textil, S.A.","base"),("Inditex","acronym"),("Zara","former_name")]),
 # ---- Industrial / aerospace / telecom ----
 "ge": ("General Electric Company","US",[("General Electric Company","base"),("General Electric Co.","abbrev"),("GE","acronym"),("Genral Electric","typo")]),
 "mmm": ("3M Company","US",[("3M Company","base"),("Minnesota Mining and Manufacturing","former_name"),("3M","suffix"),("MMM","ticker")]),
 "siemens": ("Siemens AG","DE",[("Siemens AG","base"),("Siemens Aktiengesellschaft","suffix"),("Siemens","suffix")]),
 "honeywell": ("Honeywell International Inc.","US",[("Honeywell International Inc.","base"),("Honeywell","suffix"),("HON","ticker")]),
 "caterpillar": ("Caterpillar Inc.","US",[("Caterpillar Inc.","base"),("Caterpillar","suffix"),("CAT","ticker")]),
 "boeing": ("The Boeing Company","US",[("The Boeing Company","base"),("Boeing Co","suffix"),("Boeing","the_prefix"),("BA","ticker")]),
 "airbus": ("Airbus SE","NL",[("Airbus SE","base"),("Airbus Group","former_name"),("Airbus","suffix"),("EADS","former_name")]),
 "lockheed": ("Lockheed Martin Corporation","US",[("Lockheed Martin Corporation","base"),("Lockheed Martin","suffix"),("LMT","ticker")]),
 "rtx": ("RTX Corporation","US",[("RTX Corporation","base"),("Raytheon Technologies","former_name"),("Raytheon","abbrev")]),
 "att": ("AT&T Inc.","US",[("AT&T Inc.","base"),("AT&T","suffix"),("A T & T","spacing_punct"),("American Telephone & Telegraph","former_name")]),
 "verizon": ("Verizon Communications Inc.","US",[("Verizon Communications Inc.","base"),("Verizon","suffix"),("VZ","ticker")]),
 "vodafone": ("Vodafone Group Plc","GB",[("Vodafone Group Plc","base"),("Vodafone","suffix"),("Vodaphone","typo")]),
 "comcast": ("Comcast Corporation","US",[("Comcast Corporation","base"),("Comcast","suffix"),("CMCSA","ticker")]),
 "cisco": ("Cisco Systems, Inc.","US",[("Cisco Systems, Inc.","base"),("Cisco Systems","suffix"),("Cisco","abbrev"),("CSCO","ticker")]),
 # ---- Conglomerates / luxury / misc multinational ----
 "samsung": ("Samsung Electronics Co., Ltd.","KR",[("Samsung Electronics Co., Ltd.","base"),("Samsung Electronics","suffix"),("삼성전자","translit"),("Samsung Elec","abbrev")]),
 "sony": ("Sony Group Corporation","JP",[("Sony Group Corporation","base"),("Sony Corporation","former_name"),("Sony","suffix"),("ソニー","translit")]),
 "hitachi": ("Hitachi, Ltd.","JP",[("Hitachi, Ltd.","base"),("Hitachi","suffix"),("日立製作所","translit")]),
 "lvmh": ("LVMH Moët Hennessy Louis Vuitton","FR",[("LVMH Moët Hennessy Louis Vuitton","base"),("Moet Hennessy Louis Vuitton","translit"),("LVMH","acronym")]),
 "kering": ("Kering S.A.","FR",[("Kering S.A.","base"),("Kering","suffix"),("PPR","former_name")]),
 "hermes": ("Hermès International S.A.","FR",[("Hermès International S.A.","base"),("Hermes International","translit"),("Hermès","suffix")]),
 "maersk": ("A.P. Moller-Maersk A/S","DK",[("A.P. Moller-Maersk A/S","base"),("AP Moller Maersk","spacing_punct"),("Maersk","suffix"),("A.P. Møller – Mærsk","translit")]),
 "tcs": ("Tata Consultancy Services Limited","IN",[("Tata Consultancy Services Limited","base"),("Tata Consultancy Services","suffix"),("TCS","acronym")]),
 "abinbev": ("Anheuser-Busch InBev SA/NV","BE",[("Anheuser-Busch InBev SA/NV","base"),("AB InBev","abbrev"),("Anheuser Busch InBev","spacing_punct")]),
 # ============ hard-negative families (token collisions, diff entities) ============
 # "United"
 "united_air": ("United Airlines Holdings, Inc.","US",[("United Airlines Holdings, Inc.","base"),("United Airlines","suffix"),("United Air Lines","spacing_punct"),("UAL","ticker")]),
 "ups": ("United Parcel Service, Inc.","US",[("United Parcel Service, Inc.","base"),("United Parcel Service","suffix"),("UPS","acronym")]),
 "unitedhealth": ("UnitedHealth Group Incorporated","US",[("UnitedHealth Group Incorporated","base"),("UnitedHealth Group","suffix"),("UNH","ticker")]),
 # "American"
 "amer_air": ("American Airlines Group Inc.","US",[("American Airlines Group Inc.","base"),("American Airlines","suffix"),("AAL","ticker")]),
 "aep": ("American Electric Power Company, Inc.","US",[("American Electric Power Company, Inc.","base"),("American Electric Power","suffix"),("AEP","acronym")]),
 "aig": ("American International Group, Inc.","US",[("American International Group, Inc.","base"),("American International Group","suffix"),("AIG","acronym")]),
 # "General"
 "genmills": ("General Mills, Inc.","US",[("General Mills, Inc.","base"),("General Mills","suffix"),("GIS","ticker")]),
 "gendynamics": ("General Dynamics Corporation","US",[("General Dynamics Corporation","base"),("General Dynamics","suffix"),("GD","ticker")]),
 # "Standard"
 "stanchart": ("Standard Chartered PLC","GB",[("Standard Chartered PLC","base"),("Standard Chartered","suffix")]),
 "sandp": ("S&P Global Inc.","US",[("S&P Global Inc.","base"),("Standard & Poor's","former_name"),("S&P Global","suffix")]),
 # "Bank of"
 "bankofchina": ("Bank of China Limited","CN",[("Bank of China Limited","base"),("Bank of China","suffix"),("中国银行","translit")]),
 "bankofmontreal": ("Bank of Montreal","CA",[("Bank of Montreal","base"),("BMO","acronym"),("BMO Financial Group","suffix")]),
 # "Deutsche"
 "deutschetelekom": ("Deutsche Telekom AG","DE",[("Deutsche Telekom AG","base"),("Deutsche Telekom","suffix")]),
 "deutschepost": ("Deutsche Post AG","DE",[("Deutsche Post AG","base"),("Deutsche Post DHL Group","suffix"),("DHL Group","former_name")]),
 # "Royal"
 "rbc": ("Royal Bank of Canada","CA",[("Royal Bank of Canada","base"),("RBC","acronym"),("Royal Bank","abbrev")]),
 "rbs": ("NatWest Group plc","GB",[("NatWest Group plc","base"),("Royal Bank of Scotland Group","former_name"),("RBS","acronym")]),
 # "Credit"
 "creditsuisse": ("Credit Suisse Group AG","CH",[("Credit Suisse Group AG","base"),("Credit Suisse","suffix"),("Crédit Suisse","translit")]),
 "creditagricole": ("Crédit Agricole S.A.","FR",[("Crédit Agricole S.A.","base"),("Credit Agricole","translit"),("Crédit Agricole","suffix")]),
 # "Johnson"
 "johnson_controls": ("Johnson Controls International plc","IE",[("Johnson Controls International plc","base"),("Johnson Controls","suffix")]),
 "scjohnson": ("S. C. Johnson & Son, Inc.","US",[("S. C. Johnson & Son, Inc.","base"),("SC Johnson","spacing_punct")]),
 # "Tata" group siblings
 "tatamotors": ("Tata Motors Limited","IN",[("Tata Motors Limited","base"),("Tata Motors","suffix")]),
 "tatasteel": ("Tata Steel Limited","IN",[("Tata Steel Limited","base"),("Tata Steel","suffix")]),
 # "Samsung" group siblings
 "samsung_heavy": ("Samsung Heavy Industries Co., Ltd.","KR",[("Samsung Heavy Industries Co., Ltd.","base"),("Samsung Heavy Industries","suffix")]),
 # "Delta"
 "delta_air": ("Delta Air Lines, Inc.","US",[("Delta Air Lines, Inc.","base"),("Delta Air Lines","suffix"),("DAL","ticker")]),
 "delta_elec": ("Delta Electronics, Inc.","TW",[("Delta Electronics, Inc.","base"),("Delta Electronics","suffix")]),
 # "Lincoln"
 "lincoln_fin": ("Lincoln National Corporation","US",[("Lincoln National Corporation","base"),("Lincoln Financial Group","suffix")]),
 "lincoln_elec": ("Lincoln Electric Holdings, Inc.","US",[("Lincoln Electric Holdings, Inc.","base"),("Lincoln Electric","suffix")]),
 # "Allianz" vs "Alliance"
 "allianz": ("Allianz SE","DE",[("Allianz SE","base"),("Allianz","suffix")]),
 "alliancebernstein": ("AllianceBernstein Holding L.P.","US",[("AllianceBernstein Holding L.P.","base"),("Alliance Bernstein","spacing_punct"),("AllianceBernstein","suffix")]),
 # "First"
 "firstsolar": ("First Solar, Inc.","US",[("First Solar, Inc.","base"),("First Solar","suffix"),("FSLR","ticker")]),
 "firstrepublic": ("First Republic Bank","US",[("First Republic Bank","base"),("First Republic","suffix")]),
 # "HP" split
 "hp": ("HP Inc.","US",[("HP Inc.","base"),("Hewlett-Packard","former_name"),("Hewlett Packard","spacing_punct"),("HPQ","ticker")]),
 "hpe": ("Hewlett Packard Enterprise Company","US",[("Hewlett Packard Enterprise Company","base"),("Hewlett Packard Enterprise","suffix"),("HPE","acronym")]),
}

# legal forms + connectives stripped during normalization for blocking/pairing
LEGAL = {"inc","inc.","incorporated","corp","corp.","corporation","co","co.","company",
 "ltd","ltd.","limited","llc","plc","p.l.c.","sa","s.a.","ag","se","nv","n.v.","sa/nv",
 "a/s","l.p.","lp","holdings","holding","group","the","kk","k.k.","pjsc","oao","asa",
 "kgaa","aktiengesellschaft","companies","and","of","for","und","et","des","de","la","von","y"}

def norm_tokens(s):
    s = s.lower().replace("&"," and ")
    s = re.sub(r"[^\w\s]"," ", s, flags=re.UNICODE)
    return [t for t in s.split() if t and t not in LEGAL]

# Dropped by default to land near ~120 clusters (still eyeball-able).
# These are "easy" standalone companies; every hard-negative collision family is
# kept. Re-include any by removing it from DROP (or pass --keep-all).
DROP = {
 "adobe","servicenow","uber","airbnb","paypal","block","snowflake","palantir",
 "spotify","dell","lenovo","qualcomm","broadcom","ti","micron","sanofi","bayer",
 "lilly","abbvie","bms","moderna","amgen","nissan","hyundai","porsche","renault",
 "conoco","equinor","petrobras","wells","barclays","bnp","blackrock","target",
 "lowes","adidas","starbucks","inditex","honeywell","caterpillar","lockheed",
 "vodafone","comcast","kering","hermes",
}

def build(max_clusters=None, seed=42, keep_all=False):
    rng = random.Random(seed)
    items = [(k,v) for k,v in CLUSTERS.items() if keep_all or k not in DROP]
    if max_clusters:
        items = items[:max_clusters]
    records, rid = [], 0
    for cid,(canon,country,variants) in items:
        for (v,vt) in variants:
            rid += 1
            records.append((f"r{rid:04d}", cid, canon, v, country, vt))

    by_cluster = {}
    for r in records:
        by_cluster.setdefault(r[1], []).append(r)

    # positives: all within-cluster combos
    positives = [(a,b) for rs in by_cluster.values() for a,b in itertools.combinations(rs,2)]

    # hard negatives: cross-cluster pairs sharing a content token (len>=3)
    id2 = {r[0]: r for r in records}
    tok_index = {}
    for r in records:
        for t in set(norm_tokens(r[3])):
            if len(t) >= 3:
                tok_index.setdefault(t, set()).add(r[0])
    hard = set()
    for t, rids in tok_index.items():
        rids = list(rids)
        for i in range(len(rids)):
            for j in range(i+1, len(rids)):
                a, b = id2[rids[i]], id2[rids[j]]
                if a[1] != b[1]:
                    hard.add(tuple(sorted((a[0], b[0]))))
    hard_negatives = [(id2[a], id2[b]) for a,b in hard]
    rng.shuffle(hard_negatives)

    # easy negatives: random cross-cluster, no shared token
    easy, seen, attempts = [], set(), 0
    target = min(len(positives), 220)
    while len(easy) < target and attempts < 60000:
        attempts += 1
        a, b = rng.choice(records), rng.choice(records)
        if a[1] == b[1]: continue
        if set(norm_tokens(a[3])) & set(norm_tokens(b[3])): continue
        k = tuple(sorted((a[0], b[0])))
        if k in seen: continue
        seen.add(k); easy.append((a,b))

    # cap hard negatives near positive count to keep balance
    hard_negatives = hard_negatives[:min(len(hard_negatives), int(len(positives)*0.8))]

    pairs = []
    for a,b in positives:       pairs.append([a[3],b[3],1,"positive",a[1],b[1]])
    for a,b in hard_negatives:  pairs.append([a[3],b[3],0,"hard_negative",a[1],b[1]])
    for a,b in easy:            pairs.append([a[3],b[3],0,"easy_negative",a[1],b[1]])
    rng.shuffle(pairs)
    pairs = [[f"p{i:04d}",*p] for i,p in enumerate(pairs,1)]
    return records, pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--max-clusters", type=int, default=None)
    ap.add_argument("--keep-all", action="store_true", help="include the DROP set too (~159 clusters)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    records, pairs = build(a.max_clusters, a.seed, a.keep_all)

    with open(os.path.join(a.outdir,"company_records.csv"),"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["record_id","cluster_id","canonical_name","name_variant","country","variation_type"]); w.writerows(records)
    with open(os.path.join(a.outdir,"company_pairs.csv"),"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["pair_id","name_a","name_b","label","pair_type","cluster_a","cluster_b"]); w.writerows(pairs)

    npos = sum(1 for p in pairs if p[3]==1)
    nhard = sum(1 for p in pairs if p[4]=="hard_negative")
    neasy = sum(1 for p in pairs if p[4]=="easy_negative")
    nclu = len({r[1] for r in records})
    print(f"clusters={nclu}  records={len(records)}  pairs={len(pairs)} (pos={npos} hard={nhard} easy={neasy})")
    print(f"written to {a.outdir}/company_records.csv and {a.outdir}/company_pairs.csv")

if __name__ == "__main__":
    main()
