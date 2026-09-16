# Visual slices draft report

## Counts

- table_lookup: 10 questions produced (target 10), spread across 10 distinct documents.
- diagram_lookup: 8 questions produced (target 8), spread across 8 distinct documents.
- Total: 18/18. No padding was needed; both slices hit target with genuinely confirmed tables/diagrams.
- Language mix: table_lookup is 5 English / 5 Italian. diagram_lookup is 3 English / 5 Italian. Overall 8 English / 10 Italian across the set.

## Documents and pages used

table_lookup:
- Arduino Opta PLC Datasheet, p.8 (Power Specification table, max permissible voltage)
- ABB ACS550-01/U1 User's Manual, p.20 (flange mounting kit table)
- ABB MNS Switchgear manual, p.15 (connection unit torque table)
- Interroll 24V Roller Conveyor manual, p.20 (roller merge dimensions table)
- Bonfiglioli Active Solution Drive catalog, p.20 (ACT401 Dati tecnici table)
- Mecc Alte DSR technical guide, p.15 (Table 7, fourth status word values)
- OMAL butterfly valve manual, p.5 (screw tightening torque table)
- Baltur TBG 80-85 LX-P scheda tecnica, p.6 (vapor pressure probe table)
- FAAC Catalogo Sicurezza 2022, p.30 (XKPR reader technical characteristics table)
- Riello burner manual, p.16 (Tab. G, boiler plate drilling dimensions)

diagram_lookup:
- Arduino Opta PLC Datasheet, p.9 (Product View diagram, callout 3J terminal labels)
- Danfoss VLT AutomationDrive operating guide, p.20 (GLCP labeled diagram, key 9)
- Bray Butterfly Valves manual, p.15 (Series 20/21 exploded view, item 4)
- Legrand KEOR COMPACT manual, p.20 (rear view diagram, callout 17)
- Bonfiglioli Active Solution Drive catalog, p.40 (encoder module terminal diagram, X410B.5)
- Riello burner manual, p.20 (Fig. 16 gas supply line schematic, item 4)
- Mecc Alte DSR technical guide, p.8 (regulator wiring diagram, wire color at U2)
- OMAL butterfly valve manual, p.8 (exploded view, item 5)

Every entry above was visually confirmed by opening the actual page-N.png image and reading the table or diagram directly (not inferred from extracted text), per the task's grounding requirement. Several pages I visited turned out to be text-only, safety-instruction pages, or pressure/flow charts too continuous to pin an exact value to, and were rejected as candidates.

## How visual-heavy the corpus is

I opened roughly 55 page images across 20 of the 28 documents to land on these 18 questions. Rough hit rate: about 1 in 3 pages sampled contained a genuine, unambiguous table or diagram suitable for grounding a question (as opposed to pure prose, warning/safety text, or a continuous chart curve that's hard to pin to one exact value). Electrical/mechanical equipment manuals (drives, switchgear, PLCs, valves, burners, UPS units) were reliably rich in labeled diagrams and spec tables; pure safety-instruction manuals (3M respirator, parts of the compressor manuals) were mostly text and were skipped. Exploded-parts diagrams with numbered callouts plus an adjoining legend table (Bray, OMAL, Legrand, Bonfiglioli encoder module) were the most common and reliable diagram_lookup source; dense numeric spec/rating/dimension tables (ABB, Arduino, Bonfiglioli, Baltur, Riello, Mecc Alte) were the most reliable table_lookup source.

No em dash character was used anywhere in this report or in the JSON draft.
