# Proximal Policy Optimization Algorithms (PPO)

[arXiv:1707.06347](https://arxiv.org/abs/1707.06347) · Schulman, Wolski, Dhariwal, Radford, Klimov · 2017 · BibTeX: `ppo2017`

## Problém

REINFORCE odhaduje gradient ako

$$\hat g = \hat{\mathbb{E}}_t\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \, \hat A_t\right],$$

kde $\hat{\mathbb{E}}_t$ je priemerovanie cez vzorky a $\hat A_t$ je odhad výhody. V praxi sa $\hat g$ získa derivovaním funkcie straty

$$L^{PG}(\theta) = \hat{\mathbb{E}}_t\left[\log \pi_\theta(a_t \mid s_t) \, \hat A_t\right].$$

Algoritmus je nestabilný: jeden veľký krok gradientu odsunie politiku ďaleko od aktuálnej a kvalita sa zrúti. TRPO to obmedzuje explicitnou podmienkou na KL-divergenciu medzi novou a starou politikou, ale vyžaduje optimalizáciu druhého rádu, čo je drahé a komplikované na veľkých sieťach.

PPO ponúka aproximáciu prvého rádu pre rovnaký cieľ.

## Metóda

Pomer pravdepodobností novej a starej politiky:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}.$$

Orezaná surrogate funkcia straty:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta) \hat A_t, \; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat A_t\right)\right].$$

Malé $\epsilon$ obmedzuje, ako ďaleko sa môže politika posunúť v jednom kroku. Orezávanie nuluje gradient len vtedy, keď pomer opúšťa okno **v smere zlepšenia cieľa**: pri $\hat A_t > 0$ a $r_t > 1 + \epsilon$ (aktér chce ešte viac zvýšiť pravdepodobnosť už dobrej akcie) alebo pri $\hat A_t < 0$ a $r_t < 1 - \epsilon$ (aktér chce ešte viac znížiť pravdepodobnosť už zlej akcie). V opačnom smere sa gradient nenuluje — politika môže stále korigovať chyby.

Vonkajšie $\min$ dáva pesimistický dolný odhad: pre každý krok sa berie menšia z reálneho a orezaného pomeru, čo nedovoľuje precenenie zlepšenia cieľa ani pri kladnej, ani pri zápornej výhode.

Výhoda sa odhaduje cez GAE (Schulman et al. 2016):

$$\hat A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \ldots + (\gamma\lambda)^{T-t-1}\delta_{T-1}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).$$

$\lambda$ vyvažuje skreslenie a rozptyl: pri $\lambda = 0$ je to čistý TD-target (nízky rozptyl, vysoké skreslenie), pri $\lambda = 1$ je to Monte-Carlo-podobný návrat na úplnej trajektórii; na skrátenom segmente môže pri $\lambda = 1$ na konci stále byť prítomný bootstrap $V(s_T)$.

PPO robí niekoľko epoch aktualizácií na rovnakom dávke skúseností. Orezávanie zabezpečuje, že po niekoľkých epochách politika nezájde príliš ďaleko.

Úplná funkcia straty kombinuje aktéra, kritika a entropický bonus:

$$L^{CLIP+VF+S}_t(\theta) = \hat{\mathbb{E}}_t\left[L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t)\right],$$

kde $L^{VF}_t$ je kvadratická strata pre value-funkciu, $S[\pi_\theta](s_t)$ je entropia politiky v stave $s_t$, a $c_1$, $c_2$ sú váhové koeficienty.

## Výsledky

Na MuJoCo (Hopper, Walker, HalfCheetah) PPO dosahuje úroveň TRPO pri oveľa nižšej zložitosti implementácie. Na Atari článok porovnáva s A2C a ACER: PPO vyhráva v priemernej tréningovej odmene na 30 hrách, ACER vyhráva v priemernej odmene za posledných 100 epizód na 28 hrách proti 19 pre PPO.

PPO sa stalo mainstreamom: OpenAI Five sa trénovala s PPO, InstructGPT a SCAR používajú PPO ako vonkajšiu slučku v RLHF.

## Obmedzenia

PPO je on-policy: každá aktualizácia vyžaduje čerstvé prechody z aktuálnej politiky. Off-policy metódy (SAC, TD3) sú efektívnejšie z pohľadu počtu interakcií na drahých prostrediach. Orezaný cieľ je aproximácia prvého rádu trust region, a na úlohách s úzkou oblasťou optimálnej politiky môže byť PPO nestabilné.

## Súvislosť s mojou prácou

Môj actor-critic je REINFORCE s baseline bez orezávania. PPO je prirodzeným rozšírením pre dlhšie horizonty, a prechod naň urobí moju inštaláciu porovnateľnou so SCAR, kde PPO je vonkajšia slučka s per-token Shapley odmenami.