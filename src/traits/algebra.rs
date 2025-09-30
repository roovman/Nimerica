/// A Magma has a binary operation that is closed.
pub trait Magma: Sized {
    fn op(&self, rhs: &Self) -> Self;
}

/// A Semigroup is a Magma with associativity.
/// (We cannot enforce associativity in Rust at compile time,
/// but we rely on implementor correctness.)
pub trait Semigroup: Magma {}

/// A Monoid is a Semigroup with an identity element.
pub trait Monoid: Semigroup {
    fn identity() -> Self;
}

/// A Group is a Monoid with inverses.
pub trait Group: Monoid {
    fn inverse(&self) -> Self;
}

/// An Abelian group is a Group with commutative op.
pub trait AbelianGroup: Group {}

/// A Ring: (R, +, *) 
/// - additive Abelian group
/// - multiplicative semigroup
/// - distributivity links them
pub trait Ring: AbelianGroup {
    fn mul(&self, rhs: &Self) -> Self;
}

/// A CommutativeRing is a Ring with commutative multiplication.
pub trait CommutativeRing: Ring {}

/// A Field is a commutative ring where every nonzero has an inverse.
pub trait Field: CommutativeRing {
    fn div(&self, rhs: &Self) -> Self;
    fn inv(&self) -> Self;
}
