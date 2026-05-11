# The Shapley Value in Machine Learning

[arXiv:2202.05594](https://arxiv.org/abs/2202.05594) · Rozemberczki, Watson, Bayer, Yang, Kiss, Nilsson, Sarkar · 2022 (IJCAI survey track) · BibTeX: `shapleyml2022`

## Teoretický základ

Hráči $N = \{1, \ldots, n\}$, hodnotová funkcia $v: 2^N \to \mathbb{R}$ s $v(\emptyset) = 0$. Shapleyho hodnota hráča $i$:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(n-|S|-1)!}{n!}\,[v(S \cup \{i\}) - v(S)].$$

Váhy $\frac{|S|!(n-|S|-1)!}{n!}$ sú pravdepodobnosťou, že $S$ je množina hráčov, ktorí prišli pred $i$ v náhodnej rovnomernej permutácii. To je „spravodlivý podiel" $i$ z $v(N) - v(\emptyset)$.

## Štyri axiómy

Shapleyho hodnota je jediná distribúcia, ktorá súčasne spĺňa:

**Efficiency**: $\sum_{i \in N} \phi_i(v) = v(N) - v(\emptyset)$.

**Symmetry**: ak $v(S \cup \{i\}) = v(S \cup \{j\})$ pre všetky $S$, tak $\phi_i = \phi_j$. Hráči nerozlíšiteľní podľa príspevku dostávajú rovnaký kredit.

**Dummy (null) player**: ak $v(S \cup \{i\}) = v(S)$ pre všetky $S$, tak $\phi_i = 0$. Hráč bez efektu dostáva nulu.

**Linearity / Additivity**: pre dve hry $v_1, v_2$ na rovnakej množine hráčov, $\phi_i(v_1 + v_2) = \phi_i(v_1) + \phi_i(v_2)$.

Prehľad zdôrazňuje, že tieto vlastnosti charakterizujú Shapleyho hodnotu — žiadne iné rozdelenie nespĺňa všetky štyri.

## Aplikácie v ML

Prehľad pokrýva feature selection, data valuation, federated learning (príspevok data silos), explainable ML (vysvetlenie jednotlivých predikcií cez rodinu SHAP), multi-agent RL (rozdelenie globálnej odmeny medzi agentov) a model valuation v ensembles.

RL sa rozoberá len v multi-agent variante, kde hráči sú agenti. Single-agent credit assignment na úrovni jednotlivých akcií jednej trajektórie sa medzi uvedenými aplikáciami nespomína.

## Výpočtové metódy

Presný výpočet vyžaduje $2^n$ vyhodnotení $v$, čo sa stáva nepraktickým už pri desiatkach hráčov. Prehľad rozoberá tri metódy aproximácie.

Monte Carlo cez permutácie — univerzálny odhadovač, $O(M \cdot n)$ zložitosť, nezaujatý. Všeobecná formulácia pre kooperatívne hry je z Castro et al. 2009; v feature attribution blízky prístup dáva Štrumbelj & Kononenko 2014.

Multilinear extension — prechod od súčtov cez permutácie k integrálom cez pravdepodobnostné distribúcie, s aproximáciou cez vzorkovanie.

Linear regression approximation — metóda KernelSHAP (Lundberg & Lee 2017), riešiaca váženú úlohu najmenších štvorcov na podvzorke koalícií.

## Súvislosť s mojou prácou

Štyri axiómy (efficiency, symmetry, dummy player, additivity) overujem testami na svojich Shapley odhadovačoch. Aplikácia v single-agent credit assignment je nastavenie, ktoré v prehľade aplikácií chýba.