# SCAR: Shapley Credit Assignment for More Efficient RLHF

[arXiv:2505.20417](https://arxiv.org/abs/2505.20417) · 2025 · BibTeX: `scar2025`

## Problém

V RLHF reward model produkuje skalárnu hodnotu pre celú vygenerovanú sekvenciu. Jeden skalár na dlhú sekvenciu tokenov dáva každému tokenu rovnaký signál v gradiente — vysoký rozptyl, pomalá konvergencia, slabé rozlíšenie dôležitých tokenov od výplňových.

Autori traktujú generovanie ako kooperatívnu hru. Tokeny alebo ich skupiny (units v terminológii článku) sú hráči, terminálna odmena je hodnota plnej koalície. Shapleyho príspevok každej jednotky je jej spravodlivý podiel na celkovej hodnote podľa axióm efficiency, symmetry, dummy player a additivity.

## Metóda

Hodnota koalície $v(S)$ sa tvorí takto: z pôvodnej sekvencie sa zachovajú len jednotky z $S$ v ich prirodzenom poradí, chýbajúce pozície sa vyplnia prázdnymi medzerami, a získaná čiastková sekvencia $y_S$ sa pošle do reward modelu. Reward model vráti skalár — to je $v(S)$.

Presný výpočet $\phi_i$ je exponenciálny v počte jednotiek. SCAR to obchádza pomocou aproximácie Owen-hodnôt s adaptívnou segmentáciou: jednotky sa zoskupujú do zmysluplných segmentov, a Owen-hodnoty sa odhadujú na získanej hierarchii (na samotný výpočet sa používa balík SHAP). Autori uvádzajú zníženie zložitosti z exponenciálnej na kvadratickú v počte jednotiek.

SCAR nenahrádza terminálny signál úplne, ale zmiešava ho so Shapley-odmenami na každom kroku:

$$R_t = R_t^{KL} + \alpha\, R_t^{Shapley} + (1 - \alpha)\, \mathbb{1}_{t = T}\, R^{terminal}.$$

Tu $R_t^{Shapley}$ sa distribuuje cez všetky kroky ako hustý signál, $R^{terminal}$ sa vydáva len na poslednom kroku cez indikátor $\mathbb{1}_{t = T}$, a $R_t^{KL}$ je obvyklá pre RLHF KL-pokuta proti referenčnému modelu. Koeficient $\alpha$ reguluje rovnováhu medzi hustým Shapley signálom a pôvodnou terminálnou odmenou.

## Výsledky

SCAR vykazuje rýchlejšiu konvergenciu a vyššiu finálnu odmenu v porovnaní s riedkym RLHF (jeden terminálny signál) a attention-vážnými základnými variantmi.

## Obmedzenia

Autori zdôrazňujú výpočtové náklady na viacnásobné volania reward modelu na vyhodnotenie čiastkových sekvencií. Kvalita celej metódy závisí od toho, ako dobre vie reward model zmysluplne hodnotiť čiastkové sekvencie — bol trénovaný na úplných a jeho správanie na čiastkových treba overiť.

## Súvislosť s mojou prácou

Beriem rovnakú kooperatívnu hru ako v SCAR, ale hodnotu koalície počítam cez svetový model v latentnom priestore, bez samostatného reward modelu a bez prechodov LLM s nahradeniami. Na mojich krátkych epizódach je k dispozícii presné enumerácie $2^T$ koalícií, čo eliminuje chybu aproximácie z Owen-segmentácie.