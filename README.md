# ERGM / Maximum-Entropy: confronto solutori con NEMtropy

Esperimento riproducibile per il corso di **Complex Network Analysis** (IMT School for Advanced Studies Lucca, prof. T. Squartini).

## Contesto teorico

I modelli a massima entropia (Exponential Random Graph Models, ERGM) permettono di costruire *null model* statistici di reti reali imponendo vincoli osservati (es. sequenza dei gradi). La stima dei parametri di Lagrange avviene per massimizzazione della log-verosimiglianza, che richiede un solutore numerico.

Vallarano *et al.* (2021) confrontano tre famiglie di solutori implementati nella libreria **NEMtropy**:

| Metodo | Descrizione |
|---|---|
| `newton` | Newton-Raphson con Hessiana piena |
| `quasinewton` | Newton-Raphson con Hessiana diagonale (approx.) |
| `fixed-point` | Iterazione a punto fisso |

I modelli testati sono:
- **UBCM** – Undirected Binary Configuration Model (`cm_exp`)
- **DBCM** – Directed Binary Configuration Model (`dcm_exp`)

## Metodologia

1. **Karate Club di Zachary** (n=34, m=78): UBCM risolto con ciascun metodo (3 run, seed casuali).
2. **Reti sintetiche G(n, p=0.05)**: grafi Erdős–Rényi non diretti (UBCM) e diretti (DBCM) con n ∈ {50, 100, 200}, 3 run per combinazione metodo × taglia.
3. Per ogni run si misura il **tempo di esecuzione** (wall-clock) e il **massimo errore relativo** sui vincoli di grado ricostruiti rispetto a quelli osservati.

L'errore relativo è definito come:

$$\varepsilon = \max_{i:\, k_i > 0} \frac{|k_i - \hat{k}_i|}{k_i}$$

dove $k_i$ è il grado osservato e $\hat{k}_i$ quello atteso dal modello stimato.

## Risultati

Tutti e tre i solutori convergono con errori relativi nell'ordine di $10^{-8}$–$10^{-15}$ (DBCM) e $10^{-8}$–$10^{-10}$ (UBCM, newton/quasinewton). Il metodo `fixed-point` su reti sparse di piccola taglia può presentare convergenza meno precisa per UBCM.

| Rete | Modello | Metodo | n | Runtime medio (s) | Max err. relativo |
|---|---|---|---|---|---|
| Karate Club | UBCM | newton | 34 | 0.752 | 6.3 × 10⁻⁹ |
| Karate Club | UBCM | quasinewton | 34 | 0.088 | 2.6 × 10⁻⁸ |
| Karate Club | UBCM | fixed-point | 34 | 0.154 | 3.3 × 10⁻⁹ |
| G(n,p) undir. | UBCM | newton | 200 | 0.003 | 1.8 × 10⁻⁸ |
| G(n,p) undir. | UBCM | quasinewton | 200 | 0.020 | 2.7 × 10⁻⁸ |
| G(n,p) undir. | UBCM | fixed-point | 200 | 0.002 | 2.2 × 10⁻⁹ |
| G(n,p) dir. | DBCM | newton | 200 | 0.024 | 5.7 × 10⁻⁷ |
| G(n,p) dir. | DBCM | quasinewton | 200 | 0.013 | 9.6 × 10⁻¹² |
| G(n,p) dir. | DBCM | fixed-point | 200 | 0.014 | 6.1 × 10⁻¹⁰ |

> Tabella completa: `results/tables/benchmark.csv` (generata dallo script).

## Quickstart

```bash
# 1) Crea venv
python3.11 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

# 2) Installa dipendenze
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3) Esegui esperimento
python src/experiment.py --outdir results --seed 42 --runs 3
```

### Fallback (se `pip install NEMtropy` fallisce)

```bash
pip install "git+https://github.com/nicoloval/NEMtropy.git"
# oppure: conda env create -f environment.yml && conda activate cna-nemtropy
```

## Output

Lo script produce:

```
results/
├── figures/
│   ├── runtime_vs_n.png
│   ├── karate_runtime_methods.png
│   └── error_heatmap.png
├── tables/
│   ├── benchmark.csv
│   └── metadata.json
└── samples/                       # (opzionale, con --sample_n > 0)
```

## Riproducibilità

- Seed globale: `--seed 42` (default). Il generatore NumPy deriva seed interni per ogni run.
- `metadata.json` registra versione Python, piattaforma e parametri.
- Per freeze delle dipendenze: `pip freeze > pip_freeze.txt`

## Struttura del repository

```
.
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── src/
│   └── experiment.py              # Script end-to-end
├── data/
│   ├── download_data.sh           # Scarica MovieLens 100K (opzionale)
│   └── raw/                       # Dati grezzi (non committati)
└── results/
    ├── figures/
    ├── tables/
    └── samples/
```

## Riferimenti

1. Vallarano, N., Bruno, M., Marchese, E., Trapani, G., Saracco, F., Cimini, G., Zanon, M. & Squartini, T. (2021). *Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints.* Scientific Reports **11**, 15227. [doi:10.1038/s41598-021-93830-4](https://doi.org/10.1038/s41598-021-93830-4) — [arXiv:2101.12625](https://arxiv.org/abs/2101.12625)
2. Squartini, T. & Garlaschelli, D. (2011). *Analytical maximum-likelihood method to detect patterns in real networks.* New Journal of Physics **13**, 083001. [doi:10.1088/1367-2630/13/8/083001](https://doi.org/10.1088/1367-2630/13/8/083001)
3. Saracco, F., Di Clemente, R., Gabrielli, A. & Squartini, T. (2015). *Randomizing bipartite networks: the case of the World Trade Web.* Scientific Reports **5**, 10595. [doi:10.1038/srep10595](https://doi.org/10.1038/srep10595)
4. Zachary, W. W. (1977). *An information flow model for conflict and fission in small groups.* Journal of Anthropological Research **33**(4), 452–473.
5. NEMtropy (Maximum Entropy Hub, IMT Lucca): [github.com/nicoloval/NEMtropy](https://github.com/nicoloval/NEMtropy) — [PyPI](https://pypi.org/project/NEMtropy/)

## Licenza

Progetto didattico — nessuna licenza specifica.
