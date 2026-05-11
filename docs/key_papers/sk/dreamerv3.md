# Mastering Diverse Domains through World Models (DreamerV3)

[arXiv:2301.04104](https://arxiv.org/abs/2301.04104) · Hafner, Pasukonis, Ba, Lillicrap · 2023 · BibTeX: `dreamerv3`

## Problém

Model-based RL dlho sľubuje efektivitu v počte interakcií s prostredím, ale predtým každá doména vyžadovala ručné ladenie hyperparametrov; algoritmus dobrý na jednom benchmarku zaostával na inom. V3 tvrdí, že jeden a ten istý súbor hyperparametrov funguje na širokom spektre domén od Atari a DMC po Minecraft a navigation, a potvrdzuje to rozsiahlou experimentálnou časťou.

## Metóda

Architektonicky V3 pokračuje v línii RSSM (Recurrent State-Space Model). Latentný stav má dve časti: deterministickú časť $h_t$ aktualizovanú GRU, plus diskrétnu stochastickú časť $z_t$, vzorkovanú zo softmax rozdelení so straight-through gradients — gradient prechádza cez diskrétnu vzorku priamo, ako keby bola spojitá. Encoder dáva posteriórne $q(z_t \mid h_t, o_t)$ z pozorovania, dynamika dáva priorné $p(z_t \mid h_{t-1}, a_{t-1})$ bez pozorovania.

V3 používa **KL balancing aj free bits súčasne**. KL balancing je obojstranná váhová schéma pre KL-členy v ELBO, ktorá nedovolí ani posteriornému ani priornému dominovať. Free bits klipujú členy dynamiky a reprezentácie zdola konštantným prahom, a keď je KL už dobre minimalizovaná, tieto členy sa vypnú — učenie sa sústreďuje na predikciu.

Odmena prechádza cez **symlog**:

$$\mathrm{symlog}(x) = \mathrm{sign}(x) \ln(|x| + 1), \quad \mathrm{symexp}(x) = \mathrm{sign}(x)(\exp(|x|) - 1).$$

Toto nelineárne stlačenie robí učenie necitlivým na škálu odmeny. Výstup kritika sa enkóduje cez **twohot symlog regression**: diskrétna regresia v priestore transformovanom symlogom, so softmax targetom cez dve najbližšie bin-y. To oddeľuje škálu targetu od škály gradientu a odstraňuje výbuch straty pri prvej neprázdnej odmene v epizóde.

Actor aj critic sa trénujú úplne v predstavivosti. Svetový model sa prechádza na pevný horizont, kritik sa učí cez $\lambda$-návrat, EMA-kópia kritika sa používa ako regularizátor podobný target network.

## Výsledky

V Minecraft Diamond Challenge V3-agenti nachádzajú diamond — po prvýkrát algoritmus rieši toto od nuly bez ľudských demonštrácií a bez curriculum. Na Atari 200M V3 prekonáva DreamerV2, Rainbow a IQN; výsledky na širokom spektre úloh sú dosiahnuté s pevnými hyperparametrami.

## Obmedzenia

Hlavnou prednosťou V3 je universality, a explicitných obmedzení v článku je málo. Svetový model s veľkým diskrétnym latentom je drahý na trénovanie. Ablations v článku ukazujú, že tieto komponenty (KL balancing, free bits, symlog, twohot, EMA critic regularization) podstatne ovplyvňujú stabilitu aj rýchlosť učenia.

## Súvislosť s mojou prácou

Beriem kostru V3 — svetový model plus actor-critic v predstavivosti — a odstraňujem stochastický latent, KL balancing, free bits, symlog, twohot, EMA critic regularization a continue head. Nič z toho nie je potrebné pri deterministickom LLM a binárnej odmene. Encoder namiesto CNN — Transformer nad zamrznutými Qwen embeddings; update aktéra — REINFORCE s diskrétnymi akciami namiesto reparameterization.