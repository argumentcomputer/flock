//! Multiset channels end to end.
//!
//! Two element `mult` tables carry the same `(a, b, a·b)` tuples in different
//! row orders, with NO copy constraints between them: only the channel
//! relates the two, so an accept says the channel is doing the work.
//! Exercised against a real prove/verify:
//!
//! 1. A balanced channel over a permuted multiset verifies, and the wrong
//!    verifier entry (no channel spec) rejects the proof.
//! 2. A changed tuple and a duplicated tuple (multiplicity) are rejected.
//! 3. An EXPORTED channel under a PUBLIC challenge returns the challenge and
//!    both native products, which differ.
//! 4. A public-word tuple joins a side and balances a missing gate row; a
//!    wrong public word is rejected.
//!
//! Geometry: the element-only registry at ν = 12, κ = 3 (two types) sits at
//! `M = 23`. Run with `cargo test --release -p flock-prover --test channel --
//! --ignored`.

use flock_core::channel::{
    ChannelBalance, ChannelChallenge, ChannelError, ChannelSource, ChannelSpec,
};
use flock_core::circuit::Circuit;
use flock_core::element_r1cs::{ElementTableBuilder, ElementTableType};
use flock_core::field::F128;
use flock_core::pcs::PcsParams;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_prover::challenger::FsChallenger;
use flock_prover::prover::{self, UnionElementSlotInput};
use flock_prover::schedule::{IoWord, Registry, TableType};
use flock_prover::union::UnionInstance;
use flock_prover::verifier::{self, VerifyError};
use std::sync::Arc;

const DOMAIN: &[u8] = b"flock-channel-test-v0";
const NU: usize = 12;
const KAPPA: usize = 3;
const N: usize = 20;

/// SplitMix64, the repo's test RNG convention.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f128(&mut self) -> F128 {
        F128::new(self.next_u64(), self.next_u64())
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// The element `mult` gate: columns `0,1` free wires in, column `2 = z0·z1`.
fn mult_ty(kappa: usize) -> TableType {
    let mut b = ElementTableBuilder::new(kappa);
    b.free_wire(0).free_wire(1).mult(2, 0, 1);
    TableType::element(Arc::new(b.build().expect("mult block is valid"))).with_io_schema(vec![
        IoWord::input(0),
        IoWord::input(1),
        IoWord::output(2),
    ])
}

fn union_pcs_params(union: &UnionInstance<'_>) -> PcsParams {
    let k = flock_core::pcs::ligerito::embedded_initial_k_or_default(
        union.dense_m(),
        LigeritoProfile::Fast,
    );
    PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: k,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(k),
        merkle_hash: Default::default(),
    }
}

/// A `mult`-table witness holding `pairs` in row order, in the BatchMajor
/// rows-low layout.
fn table_witness(ty: &ElementTableType, pairs: &[(F128, F128)]) -> Vec<F128> {
    let at = |c: usize, j: usize| (c << NU) + j;
    let mut z = vec![F128::ZERO; ty.width() << NU];
    for (j, &(a, b)) in pairs.iter().enumerate() {
        z[at(0, j)] = a;
        z[at(1, j)] = b;
        z[at(2, j)] = a * b;
    }
    assert!(
        ty.satisfies(&z, NU, pairs.len()),
        "generated witness must satisfy"
    );
    z
}

/// The tuple a `mult` row contributes: `(a, b, a·b)`.
fn row_tuple(ty: usize) -> ChannelSource {
    ChannelSource::Gate {
        ty,
        words: vec![IoWord::input(0), IoWord::input(1), IoWord::output(2)],
    }
}

/// The native product `∏ (β + a + α·b + α²·ab)` the channel proves.
fn native_product(pairs: &[(F128, F128)], alpha: F128, beta: F128) -> F128 {
    pairs.iter().fold(F128::ONE, |acc, &(a, b)| {
        acc * (beta + a + alpha * b + alpha * alpha * (a * b))
    })
}

fn random_pairs(rng: &mut Rng, n: usize) -> Vec<(F128, F128)> {
    (0..n).map(|_| (rng.f128(), rng.f128())).collect()
}

fn shuffled(rng: &mut Rng, pairs: &[(F128, F128)]) -> Vec<(F128, F128)> {
    let mut out = pairs.to_vec();
    for i in (1..out.len()).rev() {
        out.swap(i, rng.below(i + 1));
    }
    out
}

/// One registry of two `mult` types, no wires: prove tables A and B with
/// `channels`, then verify under `verify_specs` (the same specs, normally).
struct Circuits {
    registry: Registry,
}

impl Circuits {
    fn new() -> Self {
        Self {
            registry: Registry::new(vec![mult_ty(KAPPA), mult_ty(KAPPA)], NU),
        }
    }

    #[allow(clippy::type_complexity)]
    fn prove(
        &self,
        a: &[(F128, F128)],
        b: &[(F128, F128)],
        public: &[F128],
        channels: &[ChannelSpec],
    ) -> (
        UnionInstance<'_>,
        Circuit,
        PcsParams,
        flock_core::proof::R1csProofCircuitMerged,
        flock_core::pcs::Commitment,
        flock_core::proof::UnionClassClaims,
    ) {
        let ty = self.registry.types()[0]
            .element_type()
            .expect("element type");
        let (z_a, z_b) = (table_witness(ty, a), table_witness(ty, b));
        let counts = vec![a.len(), b.len()];
        let union = UnionInstance::new(&self.registry, counts.clone());
        let pcs_params = union_pcs_params(&union);
        let circuit =
            Circuit::new(&self.registry, counts, public.len(), Vec::new()).expect("valid circuit");
        assert!(channels.iter().all(|s| s.check(&circuit)));
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claims) = prover::prove_fast_ligerito_union_circuit_with_channels(
            &union,
            &circuit,
            public,
            channels,
            &pcs_params,
            Vec::new(),
            vec![
                UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(&z_a)),
                UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(&z_b)),
            ],
            &mut ch,
        );
        (union, circuit, pcs_params, proof, commitment, claims)
    }
}

#[allow(clippy::too_many_arguments)]
fn verify(
    union: &UnionInstance<'_>,
    circuit: &Circuit,
    public: &[F128],
    channels: &[ChannelSpec],
    commitment: &flock_core::pcs::Commitment,
    proof: &flock_core::proof::R1csProofCircuitMerged,
    pcs_params: &PcsParams,
) -> Result<flock_core::proof::UnionClassClaims, VerifyError> {
    let mut ch = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_union_circuit_with_channels(
        union,
        circuit,
        public,
        channels,
        &[],
        commitment,
        proof,
        pcs_params,
        &mut ch,
    )
}

#[test]
#[ignore] // Real proofs — run with `-- --ignored`.
fn balanced_channel_over_a_permutation_verifies() {
    let circuits = Circuits::new();
    let mut rng = Rng::new(0xC4A1_1001);
    let a = random_pairs(&mut rng, N);
    let b = shuffled(&mut rng, &a);
    let spec = ChannelSpec {
        lhs: vec![row_tuple(0)],
        rhs: vec![row_tuple(1)],
        challenge: ChannelChallenge::Sampled,
        balance: ChannelBalance::Equal,
    };
    let (union, circuit, pcs_params, proof, commitment, prover_claims) =
        circuits.prove(&a, &b, &[], std::slice::from_ref(&spec));
    assert_eq!(proof.channels.len(), 1);
    assert_eq!(prover_claims.channels.len(), 1);
    let claims = verify(
        &union,
        &circuit,
        &[],
        std::slice::from_ref(&spec),
        &commitment,
        &proof,
        &pcs_params,
    )
    .expect("a permuted multiset balances");
    assert_eq!(claims.channels, prover_claims.channels);
    let out = claims.channels[0];
    assert_eq!(out.top_lhs, out.top_rhs);
    assert_eq!(out.top_lhs, native_product(&a, out.alpha, out.beta));

    // The pre-channel entry declares no channels: the proof's extra
    // argument is a statement mismatch, not something to skip.
    let mut ch = FsChallenger::new(DOMAIN);
    assert_eq!(
        verifier::verify_ligerito_union_circuit(
            &union,
            &circuit,
            &[],
            &[],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch,
        ),
        Err(VerifyError::ChannelMismatch)
    );
}

#[test]
#[ignore] // Real proofs — run with `-- --ignored`.
fn changed_and_duplicated_tuples_are_rejected() {
    let circuits = Circuits::new();
    let mut rng = Rng::new(0xC4A1_1002);
    let a = random_pairs(&mut rng, N);
    let spec = ChannelSpec {
        lhs: vec![row_tuple(0)],
        rhs: vec![row_tuple(1)],
        challenge: ChannelChallenge::Sampled,
        balance: ChannelBalance::Equal,
    };
    // One tuple replaced by a fresh one.
    let mut changed = shuffled(&mut rng, &a);
    changed[7] = (rng.f128(), rng.f128());
    let (union, circuit, pcs_params, proof, commitment, _) =
        circuits.prove(&a, &changed, &[], std::slice::from_ref(&spec));
    assert_eq!(
        verify(
            &union,
            &circuit,
            &[],
            std::slice::from_ref(&spec),
            &commitment,
            &proof,
            &pcs_params,
        ),
        Err(VerifyError::Channel(ChannelError::Unbalanced)),
        "a changed tuple must unbalance the channel"
    );
    // One tuple duplicated in place of another: the same SET, a different
    // MULTISET — the case an XOR-style check would miss.
    let mut duplicated = shuffled(&mut rng, &a);
    duplicated[3] = duplicated[4];
    let (union, circuit, pcs_params, proof, commitment, _) =
        circuits.prove(&a, &duplicated, &[], std::slice::from_ref(&spec));
    assert_eq!(
        verify(
            &union,
            &circuit,
            &[],
            std::slice::from_ref(&spec),
            &commitment,
            &proof,
            &pcs_params,
        ),
        Err(VerifyError::Channel(ChannelError::Unbalanced)),
        "a duplicated tuple must unbalance the channel"
    );
}

#[test]
#[ignore] // Real proofs — run with `-- --ignored`.
fn exported_products_under_a_public_challenge() {
    let circuits = Circuits::new();
    let mut rng = Rng::new(0xC4A1_1003);
    let a = random_pairs(&mut rng, N);
    let b = random_pairs(&mut rng, N - 3);
    let (alpha, beta) = (rng.f128(), rng.f128());
    let public = [alpha, beta];
    let spec = ChannelSpec {
        lhs: vec![row_tuple(0)],
        rhs: vec![row_tuple(1)],
        challenge: ChannelChallenge::Public { alpha: 0, beta: 1 },
        balance: ChannelBalance::Exported,
    };
    let (union, circuit, pcs_params, proof, commitment, _) =
        circuits.prove(&a, &b, &public, std::slice::from_ref(&spec));
    let claims = verify(
        &union,
        &circuit,
        &public,
        std::slice::from_ref(&spec),
        &commitment,
        &proof,
        &pcs_params,
    )
    .expect("an exported channel never requires balance");
    let out = claims.channels[0];
    assert_eq!((out.alpha, out.beta), (alpha, beta));
    assert_eq!(out.top_lhs, native_product(&a, alpha, beta));
    assert_eq!(out.top_rhs, native_product(&b, alpha, beta));
    assert_ne!(out.top_lhs, out.top_rhs);

    // The same proof under a different public challenge: the challenge words
    // are statement-bound, so this is a different statement.
    let other = [beta, alpha];
    assert!(
        verify(
            &union,
            &circuit,
            &other,
            std::slice::from_ref(&spec),
            &commitment,
            &proof,
            &pcs_params,
        )
        .is_err(),
        "a proof under one public challenge must not verify under another"
    );
}

#[test]
#[ignore] // Real proofs — run with `-- --ignored`.
fn a_public_tuple_balances_a_missing_row() {
    let circuits = Circuits::new();
    let mut rng = Rng::new(0xC4A1_1004);
    let a = random_pairs(&mut rng, N);
    // B holds every tuple of A but the first; the public segment supplies it.
    let b = shuffled(&mut rng, &a[1..]);
    let (x, y) = a[0];
    let public = [x, y, x * y];
    let spec = ChannelSpec {
        lhs: vec![row_tuple(0)],
        rhs: vec![
            row_tuple(1),
            ChannelSource::Public {
                words: vec![0, 1, 2],
            },
        ],
        challenge: ChannelChallenge::Sampled,
        balance: ChannelBalance::Equal,
    };
    let (union, circuit, pcs_params, proof, commitment, _) =
        circuits.prove(&a, &b, &public, std::slice::from_ref(&spec));
    verify(
        &union,
        &circuit,
        &public,
        std::slice::from_ref(&spec),
        &commitment,
        &proof,
        &pcs_params,
    )
    .expect("the public tuple completes the multiset");

    // A wrong public word changes the public tuple — and the statement.
    let mut wrong = public;
    wrong[2] += F128::ONE;
    assert!(
        verify(
            &union,
            &circuit,
            &wrong,
            std::slice::from_ref(&spec),
            &commitment,
            &proof,
            &pcs_params,
        )
        .is_err(),
        "a wrong public tuple must be rejected"
    );
}
