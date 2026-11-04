# References inventory (arXiv plan, C10)

Source: `docs/reference/references.md` at 1859d165. Date: 2026-10-26. Parse: every table row is one entry (112). DOI check: `api.crossref.org/works/<doi>`, one request per 1.1 s, User-Agent naming jaxfne. Verified = HTTP 200 and the listed title equal to, or a prefix of, the Crossref title (case and punctuation ignored). Raw responses: `C:/workspace/pytest_temp/l6/crossref/`. Category and figure ids are a draft mapping by the worker from the plan text (inferred; the plan gives no per-reference mapping). An entry with no figure is "not for the paper". Candidates count figure-linked entries only; a repeated DOI is counted once. Dispatcher check 2026-10-26: E023 verified (Crossref holds the main title, subtitle empty); E045 and E044 verified on DataCite (`api.datacite.org/dois/<doi>`, HTTP 200), E044's DOI added to references.md.

## Category table

| Category | Budget | Candidates | Gap |
|---|---|---|---|
| A Replicated papers | 9 | 9 | 0 |
| B Simulators and tools | 12 | 16 | -4 |
| C Plasticity and homeostasis | 14 | 6 | 8 |
| D Laminar and multi-area anatomy | 12 | 11 | 1 |
| E Fields and LFP | 8 | 15 | -7 |
| F Fitting and optimization | 9 | 2 | 7 |
| G Reproducibility and preregistration | 6 | 5 | 1 |
| H Other | 10 | 30 | -20 |
| Total | 80 | 94 | -14 |

Entries with no figure ("not for the paper"): 17. Repeated DOI (not counted): 1 (E054 = E004).

## Main table

| ID | Short cite | Category | Figures | DOI | Crossref status |
|---|---|---|---|---|---|
| E001 | Hodgkin 1952 | H | F1 | 10.1113/jphysiol.1952.sp004764 | verified |
| E002 | Izhikevich 2003 | A | F5, S9 | 10.1109/TNN.2003.820440 | verified |
| E003 | Brette 2005 | H | F1 | 10.1152/jn.00686.2005 | verified |
| E004 | Tahvili 2026 | A | F6, S11 | 10.1371/journal.pcbi.1014378 | verified |
| E005 | Wang 1996 | H | S11 | 10.1523/JNEUROSCI.16-20-06402.1996 | verified |
| E006 | Frank 1952 | E | S16 | 10.1063/1.1702037 | verified |
| E007 | Nicholson 1975 | E | S16 | 10.1152/jn.1975.38.2.356 | verified |
| E008 | Sarvas 1987 | E | S16 | 10.1088/0031-9155/32/1/004 | verified |
| E009 | Lindén 2014 | B | S16 | 10.3389/fninf.2013.00041 | verified |
| E010 | Hagen 2018 | E | S16 | 10.3389/fninf.2018.00092 | verified |
| E011 | Herreras 2016 | E | S16 | 10.3389/fncir.2016.00101 | verified |
| E012 | Pettersen 2012 | E | S16 | 10.1017/CBO9780511979958.004 | verified |
| E013 | Einevoll 2013 | E | S16 | 10.1038/nrn3599 | verified |
| E014 | Einevoll 2020 | E | S16 | 10.1007/978-1-4614-7320-6_59-2 | verified |
| E015 | Vorwerk 2014 | E | S16 | 10.1016/j.neuroimage.2014.06.040 | verified |
| E016 | FitzHugh 1961 | H | F1 | 10.1016/S0006-3495(61)86902-6 | verified |
| E017 | Nagumo 1962 | not for the paper | none | 10.1109/JRPROC.1962.288235 | verified |
| E018 | Morris 1981 | not for the paper | none | 10.1016/S0006-3495(81)84782-0 | verified |
| E019 | Wilson 1972 | H | F1 | 10.1016/S0006-3495(72)86068-5 | verified |
| E020 | Montbrió 2015 | not for the paper | none | 10.1103/PhysRevX.5.021028 | verified |
| E021 | Coombes 2005 | not for the paper | none | 10.1007/s00422-005-0574-y | verified |
| E022 | Cook 2022 | not for the paper | none | 10.46298/mna.7284 | verified |
| E023 | Ermentrout 2001 | H | S15 | 10.1016/S0896-6273(01)00178-7 | verified (Crossref main title; subtitle in the listing) |
| E024 | Muller 2018 | H | S15 | 10.1038/nrn.2018.20 | verified |
| E025 | Geweke 1982 | H | S16 | 10.1080/01621459.1982.10477803 | verified |
| E026 | Clopath 2010 | C | F7, S12 | 10.1038/nn.2479 | verified |
| E027 | Arkhipov 2018 | D | F2 | 10.1371/journal.pcbi.1006535 | verified |
| E028 | Cannon 2014 | B | F1 | 10.3389/fninf.2014.00079 | verified |
| E029 | Gleeson 2010 | B | F1 | 10.1371/journal.pcbi.1000815 | verified |
| E030 | Vella 2014 | B | F1 | 10.3389/fninf.2014.00038 | verified |
| E031 | Dura-Bernal 2019 | B | F2 | 10.7554/eLife.44494 | verified |
| E032 | Dai 2020 | B | F2 | 10.1371/journal.pcbi.1008386 | verified |
| E033 | Sanz Leon 2013 | not for the paper | none | 10.3389/fninf.2013.00010 | verified |
| E034 | Stimberg 2019 | B | F4, S6 | 10.7554/eLife.47314 | verified |
| E035 | Yavuz 2016 | B | F4 | 10.1038/srep18854 | verified |
| E036 | Vitay 2015 | B | F4 | 10.3389/fninf.2015.00019 | verified |
| E037 | Bekolay 2014 | not for the paper | none | 10.3389/fninf.2013.00048 | verified |
| E038 | Hines 1997 | B | F2 | 10.1162/neco.1997.9.6.1179 | verified |
| E039 | Gewaltig 2007 | B | F4, S6 | 10.4249/scholarpedia.1430 | verified |
| E040 | Akar 2019 | B | F4 | 10.1109/EMPDP.2019.8671560 | verified |
| E041 | Wang 2023 | B | F4 | 10.7554/eLife.86365 | verified |
| E042 | Deistler 2025 | B | F4, S6 | 10.1038/s41592-025-02895-w | verified |
| E043 | Pille 2025 | F | F7 | 10.1101/2025.11.18.689003 | verified |
| E044 | Huh 2018 | not for the paper | none | 10.48550/arXiv.1706.04698 | verified (DataCite) |
| E045 | Kidger 2021 | B | F1 | 10.48550/arXiv.2111.00254 | verified (DataCite) |
| E046 | Nosek 2015 | not for the paper | none | 10.1126/science.aab2374 | verified |
| E047 | Grant 2026 | not for the paper | none | 10.1186/s41073-026-00223-0 | verified |
| E048 | Heil 2021 | G | S7 | 10.1038/s41592-021-01256-7 | verified |
| E049 | Miłkowski 2018 | G | F8 | 10.1007/s10827-018-0702-z | verified |
| E050 | Wilkinson 2016 | G | S7 | 10.1038/sdata.2016.18 | verified |
| E051 | Rübel 2022 | G | F1 | 10.7554/eLife.78362 | verified |
| E052 | Dai 2020 | B | F1 | 10.1371/journal.pcbi.1007696 | verified |
| E053 | Crüwell 2021 | G | S15 | 10.1098/rsos.210155 | verified |
| E054 | Tahvili 2026 | A (repeat of E004) | F6, S11 | 10.1371/journal.pcbi.1014378 | verified (listed title taken from E004) |
| E055 | Tahvili 2025 | H | S11 | 10.1016/j.celrep.2025.116131 | verified |
| E056 | N'dri 2025 | A | F7, S13 | 10.1038/s41467-025-64234-z | verified |
| E057 | Nejat 2026 | A | F7, S14 | 10.1371/journal.pone.0354021 | verified |
| E058 | Destexhe 2009 | A | S16 | 10.1007/s10827-009-0164-4 | verified |
| E059 | Brunel 2000 | A | F5, S8 | 10.1023/A:1008925309027 | verified |
| E060 | Potjans 2014 | A | F6, S10 | 10.1093/cercor/bhs358 | verified |
| E061 | Vogels 2011 | A | F7, S12 | 10.1126/science.1211095 | verified |
| E062 | Mejias 2016 | A | S16 | 10.1126/sciadv.1601335 | verified |
| E063 | Batuev 2026 | H | S13 | 10.3390/dynamics6040042 | verified |
| E064 | Kopell 2000 | H | S11 | 10.1073/pnas.97.4.1867 | verified |
| E065 | Börgers 2003 | H | S11 | 10.1162/089976603321192059 | verified |
| E066 | Prinz 2004 | F | F7 | 10.1038/nn1352 | verified |
| E067 | Bastos 2012 | D | F6 | 10.1016/j.neuron.2012.10.038 | verified |
| E068 | Bastos 2015 | D | F6 | 10.1016/j.neuron.2014.12.018 | verified |
| E069 | van Kerkoerle 2014 | D | F6 | 10.1073/pnas.1402773111 | verified |
| E070 | Mendoza-Halliday 2024 | D | S16 | 10.1038/s41593-023-01554-7 | verified |
| E071 | Tiesinga 2009 | H | S11 | 10.1016/j.neuron.2009.09.009 | verified |
| E072 | Brunel 2003 | H | S11 | 10.1152/jn.01095.2002 | verified |
| E073 | van Vreeswijk 1996 | H | S8 | 10.1126/science.274.5293.1724 | verified |
| E074 | Renart 2010 | H | S8 | 10.1126/science.1179850 | verified |
| E075 | Litwin-Kumar 2012 | H | S8 | 10.1038/nn.3220 | verified |
| E076 | Ostojic 2014 | H | S8 | 10.1038/nn.3658 | verified |
| E077 | Tsodyks 1997 | H | S11 | 10.1523/jneurosci.17-11-04382.1997 | verified |
| E078 | Ozeki 2009 | H | S11 | 10.1016/j.neuron.2009.03.028 | verified |
| E079 | Litwin-Kumar 2016 | H | S11 | 10.1152/jn.00732.2015 | verified |
| E080 | Hertäg 2020 | C | S13 | 10.7554/elife.57541 | verified |
| E081 | Mazzoni 2015 | E | S16 | 10.1371/journal.pcbi.1004584 | verified |
| E082 | Mazzoni 2008 | E | S16 | 10.1371/journal.pcbi.1000239 | verified |
| E083 | Lindén 2011 | E | S16 | 10.1016/j.neuron.2011.11.006 | verified |
| E084 | Hagen 2016 | E | S16 | 10.1093/cercor/bhw237 | verified |
| E085 | Lee 2025 | H | S13 | 10.1371/journal.pcbi.1013469 | verified |
| E086 | Ness 2025 | E | S16 | 10.1371/journal.pcbi.1012303 | verified |
| E087 | Páscoa dos Santos 2025 | not for the paper | none | 10.1162/netn_a_00460 | verified |
| E088 | Vogels 2005 | H | S8 | 10.1523/jneurosci.3508-05.2005 | verified |
| E089 | Zenke 2015 | C | S12 | 10.1038/ncomms7922 | verified |
| E090 | Song 2000 | C | S12 | 10.1038/78829 | verified |
| E091 | Billeh 2020 | D | F2 | 10.1016/j.neuron.2020.01.040 | verified |
| E092 | Schmidt 2018 | D | F2 | 10.1371/journal.pcbi.1006359 | verified |
| E093 | Chaudhuri 2015 | D | S16 | 10.1016/j.neuron.2015.09.008 | verified |
| E094 | Joglekar 2018 | D | S16 | 10.1016/j.neuron.2018.02.031 | verified |
| E095 | Veit 2017 | D | S11 | 10.1038/nn.4562 | verified |
| E096 | Wilmes 2019 | C | S12 | 10.1038/s41467-019-12972-2 | verified |
| E097 | Kumar 2008 | H | S8 | 10.1523/jneurosci.2542-07.2008 | verified |
| E098 | El Boustani 2009 | not for the paper | none | 10.1162/neco.2009.02-08-710 | verified |
| E099 | di Volo 2019 | H | S11 | 10.1162/neco_a_01173 | verified |
| E100 | Spreizer 2019 | H | S15 | 10.1371/journal.pcbi.1007432 | verified |
| E101 | Rosenbaum 2017 | not for the paper | none | 10.1038/nn.4433 | verified |
| E102 | Huang 2019 | not for the paper | none | 10.1016/j.neuron.2018.11.034 | verified |
| E103 | Keane 2015 | H | S15 | 10.1523/jneurosci.1669-14.2015 | verified |
| E104 | Mejias 2012 | not for the paper | none | 10.1103/physrevlett.108.228102 | verified |
| E105 | Wang 2002 | not for the paper | none | 10.1016/s0896-6273(02)01092-9 | verified |
| E106 | Hansel 2012 | not for the paper | none | 10.1523/jneurosci.6284-11.2012 | verified |
| E107 | Wacongne 2012 | H | S13 | 10.1523/jneurosci.5003-11.2012 | verified |
| E108 | Spratling 2008 | H | S13 | 10.3389/neuro.10.004.2008 | verified |
| E109 | Rao 1999 | H | S13 | 10.1038/4580 | verified |
| E110 | Markram 2015 | D | F2 | 10.1016/j.cell.2015.09.029 | verified |
| E111 | van Vreeswijk 2025 | C | F7 | 10.1101/2025.09.04.674182 | verified |
| E112 | Jansen 1995 | E | S16 | 10.1007/BF00199471 | verified |

## Simulator and tool references named in the plan

| Tool (plan) | Status in references.md |
|---|---|
| JAX | absent: no JAX library citation (only the BrainPy and Equinox rows mention JAX) |
| Brian2 | present: E034 (Stimberg 2019) |
| NEST | present: E039 (Gewaltig and Diesmann 2007) |
| NEURON | present: E038 (Hines and Carnevale 1997) |
| Jaxley | present: E042 (Deistler 2025) |
| NetPyNE | present: E031 (Dura-Bernal 2019) |
| LFPy | present: E009 (Linden 2014), E010 (Hagen 2018, LFPy 2.0) |

