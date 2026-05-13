# Poznámky rečníka — Shapleyho credit-assignment pre algoritmus Dreamer v úlohách s terminálnou odmenou

Projektový seminár 1 · 15.05.2026 · Aleksandr Bukhtoiarov

## Slide 2 — Hlavná motivácia

Začnem s niečím, čo priamo nesúvisí s mojou prácou, ale je skôr jej dôsledkom. Cieľ, ku ktorému sa snažia dostať mnohí ľudia pracujúci s LLM, je vyžmýkať kvalitu väčšieho modelu z menšieho. Vo svojej práci to robím cez Shapleyho a Dreamer.

## Slide 3 — Formulácia: RL s terminálnou odmenou

Teraz to, čo robím. Toto je posilňované učenie vo formulácii s terminálnou odmenou: epizóda je hodnotená jedným skalárom úplne na konci, žiadna priebežná spätná väzba. Štandardný problém — ako rozdeliť tento jeden signál medzi T akcií.

## Slide 4 — Nástroj 1: Shapleyho dekompozícia

Zo SCAR 2025 beriem kooperatívno-hernú formuláciu. Akcie sú hráči, terminálna odmena je hodnota plnej koalície, Shapleyho dekompozícia dáva každému kroku jeho férový podiel. Jedno riedke R sa mení na hustý signál φᵢ pre každú akciu.

## Slide 5 — Nástroj 2: model sveta ako orákulum v(S)

Shapleyho dekompozícia potrebuje ohodnotiť hodnotu koalície v(S). V RLHF túto úlohu zohráva model odmeny natrénovaný na preference-pároch. U mňa je to model sveta z Dreamera: encoder + GRU + reward head, natrénované na rolloutoch v prostredí cez MSE proti pozorovanému R. Na counterfactual-trajektórii sa akcie mimo koalície nahrádzajú baseline-om, prejdeme v latente, reward head dáva v(S).

## Slide 6 — Agent: bunková pracovná pamäť

Nastavenie agenta je neštandardné. Nie jeden rastúci textový kontext, ale K krátkych buniek, každá je prepisovateľná pamäť. Každá akcia — LLM vyberie bunku a zapíše do nej; do jeho vlastného kontextu sa podáva len aktuálny obsah buniek. Prečo práve takto — vrátim sa k tomu na konci.

## Slide 7 — Architektúra

Zamrznutý LLM beží len pri zbere rolloutov. Encoder beží na každej bunke zvlášť, GRU aktualizuje stav pod akciou, reward head dáva skalár. Actor a critic — self-attention nad K+1 tokenmi stavu, trénované úplne v imaginácii modelu sveta cez REINFORCE.

## Slide 8 — Prvý experiment

Úloha je viackroková aritmetika so zátvorkami, LLM zamrznutý. Vanilla sotva poráža náhodnú politiku. Shapley dvíha výsledok 2.3-krát — hustý signál na bunkovej pamäti dáva prírastok.

## Slide 9 — Späť k LLM: CoT-balast

Späť k motivácii z prvého slide-u. Štandardná cesta ku kvalite je dlhý kontext alebo chain-of-thought; LLM si dáva medzivýpočty späť do vstupu. S každým krokom kontext rastie, ale reálne v ňom pracuje len malá časť — aktuálne premenné a aktívne hypotézy. Zvyšok je balast, a veľký LLM to zvláda vďaka kapacite.

## Slide 10 — Dlhodobá hypotéza: menší LLM s bunkami

V mojej formulácii kontext LLM-volania zostáva krátky — len aktuálne bunky. To dáva nádej na dlhodobú hypotézu: na úlohách s dlhým reťazcom uvažovania menší LLM s bunkovou pamäťou môže dohnať väčší LLM bez pamäte. Zdôrazňujem — toto je hypotéza a smer, nie dnešný výsledok; prvý experiment overuje len credit assignment, veľkosť LLM je zatiaľ fixná.

## Slide 11 — Najbližšie kroky

Prvé — dotiahnuť aritmetický experiment do stabilného riešenia. Súčasných 0.30 od Shapleyho je first run, bez entropy-bonusu a lr-scheduler-u; paralelne treba ground-truth-validovať φ priamym prechodom reálneho LLM. Druhé — pripojiť XGBLoRA, metódu dotrénovania LLM z mojej bakalárskej práce, aby hustý Shapleyho signál tlačil nielen na actor-a, ale na samotný LLM. Tretie — paralelne dopísať chýbajúce kapitoly práce.

## Slide 12 — Ďalej: porovnanie s konkurenčnými metódami

Keď je pipeline stabilný a LLM sa začne dotrénovávať, cieľ je obísť tri najbližšie credit-assignment metódy na porovnateľných úlohách. AgeMem trénuje agenta v šiestich memory-operáciách cez step-level GRPO. MemAct ponúka operátor Prune&Write nad bunkami a trénuje cez DCPO. RUDDER — return decomposition cez LSTM-prediktor návratu. Všetky tri sú credit assignment v epizódach s riedkou odmenou, ale bez kooperatívno-hernej formulácie a bez modelu sveta ako orákula. Dlhodobo za tým stojí hypotéza z prvého slide-u: empiricky overiť, či menší LLM s bunkovou pamäťou môže dohnať väčší bez pamäte.

## Slide 13 — Záver

Ďakujem. Kód, testy, poznámky k kľúčovým článkom v EN a SK, LaTeX zdroj — na linku.