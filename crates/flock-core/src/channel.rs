//! Multiset channels over a circuit's cell space.
//!
//! A channel is two multisets of tuples — a LEFT side and a RIGHT side — each
//! drawn from the rows of gate slots (every live row of a registry type
//! contributes the tuple formed by its listed schema words) and from public
//! words (one tuple each). The argument proves the two grand products
//!
//! ```text
//!   ∏_{lhs tuples} (β + Σ_k α^k · t_k)      ∏_{rhs tuples} (β + Σ_k α^k · t_k)
//! ```
//!
//! with the tag-free product-GKR ([`product_gkr::prove_plain_grouped`]) over
//! the channel's own grouped domain — one group of `2^ν` rows per source,
//! dead rows implicit ones — and reduces both roots to packed-direct
//! evaluation claims on the committed union polynomial, the intake the
//! wiring argument's gather claims already use. No channel column is ever
//! committed: the prover's extra work is `O(N)` field multiplications and one
//! eq-weighted row fold per tuple column.
//!
//! What a channel MEANS is the caller's. Under [`ChannelBalance::Equal`] the
//! verifier requires the two products to agree — pushes equal pulls as
//! multisets, multiplicity included. Under [`ChannelBalance::Exported`] both
//! roots are returned for an enclosing statement to consume, e.g. a product
//! that continues across many proofs.
//!
//! Soundness: distinct multisets of tuples give distinct polynomials in
//! `(α, β)` — every factor is monic in `β`, so multiplicity survives in
//! characteristic two — and two distinct products of `N` factors of `α`-degree
//! `< W` agree at a random `(α, β)` with probability at most `N·W / |F|`. The
//! challenge is either sampled from the transcript at the channel's position
//! (after the commitment, like the wiring's) or read from the public segment
//! (a challenge an enclosing statement fixed, e.g. a hash of many
//! commitments); the verifier reads the same words, so a proof under one
//! challenge cannot verify under another.
//!
//! Dummy rows: a gate source contributes its type's DECLARED rows; the rows
//! beyond the count are dead for the GKR (implicit ones) and zero in the
//! committed polynomial, so the verifier's reconstruction
//! `Σ_k α^k · ĉ_k(ρ_row) + (β+1)·live(ρ) + 1` needs only the live-mask
//! evaluation the wiring already uses. Rows a gate keeps DISABLED inside its
//! count must self-cancel: push and pull the same tuple.

use crate::challenger::Challenger;
use crate::circuit::{CellSlot, CellSpace, Circuit};
use crate::field::F128;
use crate::pcs::{DirectEqInd, PackedDirectClaim};
use crate::product_gkr::{self, BatchedGrinding, LiveMask, PlainGkrProof};
use crate::schedule::IoWord;
use crate::zerocheck::univariate_skip::build_eq;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const DOMAIN_CHANNEL: &[u8] = b"flock-channel-v0";

/// Where one side's tuples come from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelSource {
    /// Every live row of registry slot `ty`: the tuple is that row's listed
    /// schema words, in order (`words.len()` is the tuple width).
    Gate { ty: usize, words: Vec<IoWord> },
    /// One tuple of public words, by public-segment index.
    Public { words: Vec<usize> },
}

impl ChannelSource {
    /// Tuple width.
    pub fn width(&self) -> usize {
        match self {
            Self::Gate { words, .. } => words.len(),
            Self::Public { words } => words.len(),
        }
    }
}

/// Where the fingerprint challenge `(α, β)` comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelChallenge {
    /// Squeezed from the transcript at the channel's position — after the
    /// commitment and every class PIOP, like the wiring's challenge.
    Sampled,
    /// Read from the public segment at these word indices. The enclosing
    /// statement is responsible for how those words were chosen.
    Public { alpha: usize, beta: usize },
}

/// What the verifier does with the two roots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelBalance {
    /// Require `∏ lhs = ∏ rhs`.
    Equal,
    /// Return both roots; the enclosing statement consumes them.
    Exported,
}

/// One channel: verifier policy, digest-free (the verifier holds it, like
/// the circuit).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelSpec {
    pub lhs: Vec<ChannelSource>,
    pub rhs: Vec<ChannelSource>,
    pub challenge: ChannelChallenge,
    pub balance: ChannelBalance,
}

impl ChannelSpec {
    /// Groups per side: the larger side's source count rounded up to a power
    /// of two, at least one. The smaller side's missing groups are dead.
    pub fn groups(&self) -> usize {
        self.lhs
            .len()
            .max(self.rhs.len())
            .max(1)
            .next_power_of_two()
    }

    /// `μ` of the channel's product domain for a circuit with `2^nu` rows.
    pub fn mu(&self, nu: usize) -> usize {
        nu + self.groups().trailing_zeros() as usize
    }

    /// Whether the spec names only existing cell slots and public words,
    /// with non-empty tuples and a product domain of at least two entries.
    pub fn check(&self, circuit: &Circuit) -> bool {
        let cells = circuit.cells();
        let n = self.groups() << cells.nu();
        n >= 2
            && self.lhs.iter().chain(&self.rhs).all(|source| match source {
                ChannelSource::Gate { ty, words } => {
                    !words.is_empty()
                        && *ty < circuit.counts().len()
                        && words.iter().all(|w| cell_slot(cells, *ty, *w).is_some())
                }
                ChannelSource::Public { words } => {
                    !words.is_empty() && words.iter().all(|&w| w < circuit.num_public())
                }
            })
            && match self.challenge {
                ChannelChallenge::Sampled => true,
                ChannelChallenge::Public { alpha, beta } => {
                    alpha < circuit.num_public() && beta < circuit.num_public()
                }
            }
    }

    /// Number of gate columns on both sides — the gather length.
    pub fn gather_len(&self) -> usize {
        self.lhs
            .iter()
            .chain(&self.rhs)
            .map(|s| match s {
                ChannelSource::Gate { words, .. } => words.len(),
                ChannelSource::Public { .. } => 0,
            })
            .sum()
    }

    /// Live rows per group of one side.
    fn lens(side: &[ChannelSource], circuit: &Circuit, groups: usize) -> Vec<usize> {
        let rows = 1usize << circuit.cells().nu();
        (0..groups)
            .map(|g| match side.get(g) {
                Some(ChannelSource::Gate { ty, .. }) => circuit.counts()[*ty].min(rows),
                Some(ChannelSource::Public { .. }) => 1,
                None => 0,
            })
            .collect()
    }
}

/// The cell-slot index of `(ty, word)`, if the registry type has that word.
fn cell_slot(cells: &CellSpace, ty: usize, word: IoWord) -> Option<usize> {
    cells.slots()[..cells.num_gate_slots()]
        .iter()
        .position(|s| matches!(s, CellSlot::Gate { ty: t, word: w } if *t == ty && *w == word))
}

/// A channel's transcript.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelProof {
    /// The sampled challenge's PoW nonce when the policy grinds it; `None`
    /// for a public challenge or a grind-free policy.
    pub alpha_nonce: Option<u64>,
    pub gkr: PlainGkrProof,
    /// One eq-weighted row fold per gate column, LEFT sources then RIGHT, in
    /// spec order — the values of the packed-direct claims the opening binds.
    pub gather: Vec<F128>,
}

/// What a verified channel leaves for the enclosing statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelOutput {
    pub alpha: F128,
    pub beta: F128,
    pub top_lhs: F128,
    pub top_rhs: F128,
}

/// Why a channel proof was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChannelError {
    /// The spec does not fit the circuit.
    MalformedSpec,
    /// Wrong gather length or public segment.
    MalformedProof,
    /// The tag-free product-GKR rejected.
    Gkr(product_gkr::VerifyError),
    /// The gather values do not recombine to the GKR's leaf evaluations.
    Recombination,
    /// `∏ lhs ≠ ∏ rhs` under [`ChannelBalance::Equal`].
    Unbalanced,
    /// A challenge grinding witness was missing, superfluous or wrong.
    InvalidGrinding,
}

/// Prove one channel over the committed (padded) union buffer `packed` and
/// the circuit's public words. Returns the transcript, the packed-direct
/// claims for the merged opening (gate columns, LEFT then RIGHT, spec order)
/// and the challenge plus roots.
pub fn prove_channel<C: Challenger>(
    circuit: &Circuit,
    packed: &[F128],
    public: &[F128],
    spec: &ChannelSpec,
    grinding: BatchedGrinding,
    ch: &mut C,
) -> (ChannelProof, Vec<PackedDirectClaim>, ChannelOutput) {
    assert!(spec.check(circuit), "the channel spec must fit the circuit");
    assert_eq!(
        public.len(),
        circuit.num_public(),
        "public words must match the circuit's public segment"
    );
    let cells = circuit.cells();
    let nu = cells.nu();
    let rows = 1usize << nu;
    let groups = spec.groups();
    let n = groups * rows;
    let lens_l = ChannelSpec::lens(&spec.lhs, circuit, groups);
    let lens_r = ChannelSpec::lens(&spec.rhs, circuit, groups);
    let live_entries: usize = lens_l.iter().chain(&lens_r).sum();

    ch.observe_label(DOMAIN_CHANNEL);
    let (alpha, beta, alpha_nonce) = match spec.challenge {
        ChannelChallenge::Sampled => {
            let bits = grinding.fingerprint_bits_for(live_entries);
            let (nonce, alpha) = if bits != 0 {
                let (nonce, alpha) = ch.grind_pow_and_sample_f128(bits);
                (Some(nonce), alpha)
            } else {
                (None, ch.sample_f128())
            };
            (alpha, ch.sample_f128(), nonce)
        }
        ChannelChallenge::Public { alpha, beta } => {
            let (a, b) = (public[alpha], public[beta]);
            ch.observe_f128(a);
            ch.observe_f128(b);
            (a, b, None)
        }
    };

    // Leaves: `β + Σ_k α^k · t_k` on every live row; dead rows never written.
    let build = |side: &[ChannelSource], lens: &[usize]| -> Vec<F128> {
        let mut buf = crate::scratch::take_f128(n);
        assert!(buf.len() >= n);
        for (g, source) in side.iter().enumerate() {
            let base = g * rows;
            match source {
                ChannelSource::Gate { ty, words } => {
                    let cols: Vec<usize> = words
                        .iter()
                        .map(|w| {
                            let iota = cell_slot(cells, *ty, *w).expect("checked above");
                            cells.gate_word_addr(iota, 0)
                        })
                        .collect();
                    buf[base..base + lens[g]]
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(j, dst)| {
                            let mut fp = beta;
                            let mut pow = F128::ONE;
                            for &c in &cols {
                                fp += pow * packed[c + j];
                                pow *= alpha;
                            }
                            *dst = fp;
                        });
                }
                ChannelSource::Public { words } => {
                    let mut fp = beta;
                    let mut pow = F128::ONE;
                    for &w in words {
                        fp += pow * public[w];
                        pow *= alpha;
                    }
                    buf[base] = fp;
                }
            }
        }
        buf
    };
    let lhs = build(&spec.lhs, &lens_l);
    let rhs = build(&spec.rhs, &lens_r);
    let (gkr, claim) = product_gkr::prove_plain_grouped(
        lhs,
        lens_l.clone(),
        rhs,
        lens_r.clone(),
        nu,
        grinding,
        ch,
    );

    // Gather: one eq-weighted fold of the live prefix per gate column, on the
    // packed-direct claim shape (Lemma "Gather factorization").
    let eq_row = build_eq(&claim.rho[..nu]);
    let mut gather = Vec::with_capacity(spec.gather_len());
    let mut claims = Vec::with_capacity(spec.gather_len());
    for (side, lens) in [(&spec.lhs, &lens_l), (&spec.rhs, &lens_r)] {
        for (g, source) in side.iter().enumerate() {
            let ChannelSource::Gate { ty, words } = source else {
                continue;
            };
            for w in words {
                let iota = cell_slot(cells, *ty, *w).expect("checked above");
                let base = cells.gate_word_addr(iota, 0);
                let mut v = F128::ZERO;
                for (j, &e) in eq_row.iter().take(lens[g]).enumerate() {
                    v += e * packed[base + j];
                }
                let point = cells.gate_claim_point(iota, &claim.rho[..nu]);
                gather.push(v);
                claims.push(PackedDirectClaim {
                    value: v,
                    eq_ind: DirectEqInd::EqPoint(point.clone()),
                    point,
                });
            }
        }
    }
    let output = ChannelOutput {
        alpha,
        beta,
        top_lhs: gkr.top_lhs,
        top_rhs: gkr.top_rhs,
    };
    (
        ChannelProof {
            alpha_nonce,
            gkr,
            gather,
        },
        claims,
        output,
    )
}

/// Replay one channel. Returns the packed-direct `(point, value)` pairs for
/// the merged opening — in the prover's order — and the challenge plus roots.
pub fn verify_channel<C: Challenger>(
    circuit: &Circuit,
    public: &[F128],
    spec: &ChannelSpec,
    proof: &ChannelProof,
    grinding: BatchedGrinding,
    ch: &mut C,
) -> Result<(Vec<(Vec<F128>, F128)>, ChannelOutput), ChannelError> {
    if !spec.check(circuit) {
        return Err(ChannelError::MalformedSpec);
    }
    if public.len() != circuit.num_public() || proof.gather.len() != spec.gather_len() {
        return Err(ChannelError::MalformedProof);
    }
    let cells = circuit.cells();
    let nu = cells.nu();
    let groups = spec.groups();
    let mu = spec.mu(nu);
    let lens_l = ChannelSpec::lens(&spec.lhs, circuit, groups);
    let lens_r = ChannelSpec::lens(&spec.rhs, circuit, groups);
    let live_entries: usize = lens_l.iter().chain(&lens_r).sum();

    ch.observe_label(DOMAIN_CHANNEL);
    let (alpha, beta) = match spec.challenge {
        ChannelChallenge::Sampled => {
            let bits = grinding.fingerprint_bits_for(live_entries);
            let alpha = if bits != 0 {
                let nonce = proof.alpha_nonce.ok_or(ChannelError::InvalidGrinding)?;
                ch.verify_pow_and_sample_f128(nonce, bits)
                    .ok_or(ChannelError::InvalidGrinding)?
            } else {
                if proof.alpha_nonce.is_some() {
                    return Err(ChannelError::InvalidGrinding);
                }
                ch.sample_f128()
            };
            (alpha, ch.sample_f128())
        }
        ChannelChallenge::Public {
            alpha: alpha_at,
            beta: beta_at,
        } => {
            if proof.alpha_nonce.is_some() {
                return Err(ChannelError::InvalidGrinding);
            }
            let (a, b) = (public[alpha_at], public[beta_at]);
            ch.observe_f128(a);
            ch.observe_f128(b);
            (a, b)
        }
    };

    let claim =
        product_gkr::verify_plain(mu, &proof.gkr, grinding, ch).map_err(ChannelError::Gkr)?;
    if spec.balance == ChannelBalance::Equal && proof.gkr.top_lhs != proof.gkr.top_rhs {
        return Err(ChannelError::Unbalanced);
    }

    // Recombination: `L̂(ρ) = Σ_g eq(ρ_group, g)·Σ_k α^k·v_{g,k} + (β+1)·live(ρ) + 1`,
    // where a public source's fold is its tuple weighted by `eq(ρ_row, 0)`.
    let eq_row = build_eq(&claim.rho[..nu]);
    let eq_group = build_eq(&claim.rho[nu..]);
    let mut gather = proof.gather.iter();
    let mut points = Vec::with_capacity(proof.gather.len());
    let mut recombine = |side: &[ChannelSource], lens: &[usize]| -> Result<F128, ChannelError> {
        let mask = LiveMask {
            nu,
            counts: lens.to_vec(),
        };
        let mut acc = (beta + F128::ONE) * mask.live_eval(&claim.rho) + F128::ONE;
        for (g, source) in side.iter().enumerate() {
            let mut fp = F128::ZERO;
            let mut pow = F128::ONE;
            match source {
                ChannelSource::Gate { ty, words } => {
                    for w in words {
                        let v = *gather.next().ok_or(ChannelError::MalformedProof)?;
                        let iota = cell_slot(cells, *ty, *w).expect("checked above");
                        points.push((cells.gate_claim_point(iota, &claim.rho[..nu]), v));
                        fp += pow * v;
                        pow *= alpha;
                    }
                }
                ChannelSource::Public { words } => {
                    for &w in words {
                        fp += pow * public[w];
                        pow *= alpha;
                    }
                    fp *= eq_row[0];
                }
            }
            acc += eq_group[g] * fp;
        }
        Ok(acc)
    };
    let l = recombine(&spec.lhs, &lens_l)?;
    let r = recombine(&spec.rhs, &lens_r)?;
    if l != claim.l_eval || r != claim.r_eval {
        return Err(ChannelError::Recombination);
    }
    Ok((
        points,
        ChannelOutput {
            alpha,
            beta,
            top_lhs: proof.gkr.top_lhs,
            top_rhs: proof.gkr.top_rhs,
        },
    ))
}
