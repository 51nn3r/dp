# Explaining Prediction Models and Individual Predictions with Feature Contributions

[doi:10.1007/s10115-013-0679-x](https://doi.org/10.1007/s10115-013-0679-x) · Štrumbelj, Kononenko · *Knowledge and Information Systems* · 2014 · BibTeX: `strumbelj2014`

## Problém

Daný je prediktívny model $f$ (klasifikátor, regresor) a príklad $x = (x_1, \ldots, x_n)$. Cieľom je vysvetliť $f(x)$ cez príspevky jednotlivých príznakov. Metóda traktuje príznaky ako hráčov kooperatívnej hry a $f$ ako hodnotovú funkciu. Hodnota koalície $v(S)$ je teoreticky definovaná cez podmienenú strednú hodnotu $\mathbb{E}[f(X) \mid X_S = x_S]$; v praxi článok odhaduje túto hodnotu cez sampling approximation pri predpoklade nezávislosti príznakov. Shapleyho príspevok

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(n-|S|-1)!}{n!}\,[v(S \cup \{i\}) - v(S)]$$

je jednoznačne určený axiómami efficiency, symmetry, dummy player a additivity. Problém je exponenciálna zložitosť: $2^n$ vyhodnotení $v$.

## Metóda

Algoritmus Monte Carlo cez permutácie. Vzorkuje sa $M$ náhodných permutácií $\pi^{(1)}, \ldots, \pi^{(M)}$ množiny $\{1, \ldots, n\}$. Pre každú permutáciu a každý príznak $i$ sa počíta marginálny príspevok

$$\Delta_i^{(m)} = v(\text{Pre}_i^{\pi^{(m)}} \cup \{i\}) - v(\text{Pre}_i^{\pi^{(m)}}),$$

kde $\text{Pre}_i^{\pi}$ je množina príznakov idúcich pred $i$ v permutácii. Priemerovanie dáva nezaujatý odhad:

$$\hat\phi_i = \frac{1}{M}\sum_{m=1}^M \Delta_i^{(m)}.$$

Zložitosť klesá na $O(M \cdot n)$. Odhad konverguje k pravej Shapleyho hodnote pri $M \to \infty$.

Pre moju full-permutation implementáciu (keď sa každá permutácia prechádza úplne) platí vlastnosť per-permutation efficiency:

$$\sum_{i=1}^n \Delta_i^{\pi} = v(N) - v(\emptyset)$$

cez teleskopickú sumu. Axióma efficiency sa zachováva nielen v priemere, ale aj pre každú permutáciu zvlášť. Toto vyplýva zo štruktúry algoritmu, nie je to zvlášť dokazovaná vlastnosť v samotnom článku.

## Výsledky

Metóda je použiteľná na klasifikáciu aj regresiu, zohľadňuje podmnožiny, interakcie a redundancie medzi príznakmi. Stala sa štandardom pre feature attribution a základom pre SHAP (Lundberg & Lee 2017) a jeho nasledovníkov.

## Obmedzenia

Sampling approximation v článku sa opiera o predpoklad nezávislosti príznakov. V dôsledku toho je na silne korelovaných príznakoch odhad zaujatý v rozdelení kreditu medzi nimi. Rozptyl rastie s rastúcim $n$.

## Súvislosť s mojou prácou

Permutačný Monte Carlo odhadovač z tohto článku používam ako jeden z dvoch spôsobov výpočtu Shapleyho hodnôt v mojom prostredí; druhým je presné enumerácie $2^T$ koalícií, ktoré slúži ako etalón pre testy na krátkych epizódach. Konvergenciu permutačného odhadu k presnému a per-permutation efficiency overujem testami na syntetických hodnotových funkciách.