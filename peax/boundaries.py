"""Boundary geometry definitions for the environment."""

from typing import Protocol
import jax.numpy as jnp
from chex import Array
import jax


class Boundary(Protocol):
    """Protocol for boundary geometries."""

    def contains(self, position: Array) -> Array:
        """Check if a position is within the boundary.

        Args:
            position: (x, y) position to check

        Returns:
            Boolean indicating if position is inside boundary
        """
        ...

    def clip_to_boundary(self, position: Array) -> Array:
        """Clip a position to be within the boundary.

        Args:
            position: (x, y) position to clip

        Returns:
            Clipped position that is within the boundary
        """
        ...

    def sample_position(self, key: Array) -> Array:
        """Sample a random position within the boundary.

        Args:
            key: JAX random key

        Returns:
            Random position within the boundary
        """
        ...


class SquareBoundary:
    """Square boundary centered at origin."""

    def __init__(self, size: float):
        """Initialize square boundary.

        Args:
            size: Side length of the square
        """
        self.size = size
        self.half_size = size / 2.0

    def contains(self, position: Array) -> Array:
        """Check if position is within square boundary."""
        return jnp.all(jnp.abs(position) <= self.half_size)

    def clip_to_boundary(self, position: Array) -> Array:
        """Clip position to square boundary."""
        return jnp.clip(position, -self.half_size, self.half_size)

    def sample_position(self, key: Array) -> Array:
        """Sample random position in square."""
        return jax.random.uniform(
            key,
            shape=(2,),
            minval=-self.half_size,
            maxval=self.half_size
        )


class TriangleBoundary:
    """Equilateral triangle boundary centered at origin.

    The triangle has one vertex pointing up (positive y direction).
    """

    def __init__(self, size: float):
        """Initialize triangle boundary.

        Args:
            size: Side length of the equilateral triangle
        """
        self.size = size
        # Calculate vertices of equilateral triangle centered at origin
        # Height of equilateral triangle: h = (sqrt(3)/2) * side
        height = (jnp.sqrt(3.0) / 2.0) * size
        # Center of mass is at 1/3 height from base
        self.vertices = jnp.array([
            [0.0, 2.0 * height / 3.0],  # Top vertex
            [-size / 2.0, -height / 3.0],  # Bottom left
            [size / 2.0, -height / 3.0],   # Bottom right
        ])

    def _sign(self, p: Array, v1: Array, v2: Array) -> Array:
        """Helper function for point-in-triangle test."""
        return (p[0] - v2[0]) * (v1[1] - v2[1]) - (v1[0] - v2[0]) * (p[1] - v2[1])

    def contains(self, position: Array) -> Array:
        """Check if position is within triangle using barycentric coordinates."""
        v1, v2, v3 = self.vertices[0], self.vertices[1], self.vertices[2]

        d1 = self._sign(position, v1, v2)
        d2 = self._sign(position, v2, v3)
        d3 = self._sign(position, v3, v1)

        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

        return ~(has_neg & has_pos)

    def clip_to_boundary(self, position: Array) -> Array:
        """Clip position to triangle boundary.

        If outside, project to nearest edge.
        """
        # If inside, return as is
        is_inside = self.contains(position)

        def project_to_segment(p: Array, v1: Array, v2: Array) -> Array:
            """Project point to line segment."""
            segment = v2 - v1
            t = jnp.dot(p - v1, segment) / jnp.dot(segment, segment)
            t = jnp.clip(t, 0.0, 1.0)
            return v1 + t * segment

        # Project to each edge and find closest
        v1, v2, v3 = self.vertices[0], self.vertices[1], self.vertices[2]
        proj1 = project_to_segment(position, v1, v2)
        proj2 = project_to_segment(position, v2, v3)
        proj3 = project_to_segment(position, v3, v1)

        dist1 = jnp.sum((position - proj1) ** 2)
        dist2 = jnp.sum((position - proj2) ** 2)
        dist3 = jnp.sum((position - proj3) ** 2)

        # Find minimum distance projection
        min_dist = jnp.minimum(dist1, jnp.minimum(dist2, dist3))
        closest = jnp.where(
            dist1 == min_dist,
            proj1,
            jnp.where(dist2 == min_dist, proj2, proj3)
        )

        return jnp.where(is_inside, position, closest)

    def sample_position(self, key: Array) -> Array:
        """Sample random position in triangle using barycentric coordinates."""
        # Sample two uniform random variables
        r1, r2 = jax.random.uniform(key, shape=(2,))

        # Use sqrt for uniform sampling in triangle
        sqrt_r1 = jnp.sqrt(r1)
        a = 1.0 - sqrt_r1
        b = sqrt_r1 * (1.0 - r2)
        c = sqrt_r1 * r2

        # Barycentric combination of vertices
        v1, v2, v3 = self.vertices[0], self.vertices[1], self.vertices[2]
        return a * v1 + b * v2 + c * v3


class CircleBoundary:
    """Circular boundary centered at origin."""

    def __init__(self, radius: float):
        """Initialize circle boundary.

        Args:
            radius: Radius of the circle
        """
        self.radius = radius

    def contains(self, position: Array) -> Array:
        """Check if position is within circle."""
        return jnp.sum(position ** 2) <= self.radius ** 2

    def clip_to_boundary(self, position: Array) -> Array:
        """Clip position to circle boundary."""
        distance = jnp.sqrt(jnp.sum(position ** 2))
        return jnp.where(
            distance <= self.radius,
            position,
            position * (self.radius / distance)
        )

    def sample_position(self, key: Array) -> Array:
        """Sample random position in circle."""
        # Sample radius and angle
        key1, key2 = jax.random.split(key)
        r = jnp.sqrt(jax.random.uniform(key1)) * self.radius
        theta = jax.random.uniform(key2, minval=0.0, maxval=2.0 * jnp.pi)
        return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])


def create_boundary(boundary_type: str, size: float) -> Boundary:
    """Factory function to create boundary objects.

    Args:
        boundary_type: Type of boundary ("square", "triangle", or "circle")
        size: Size parameter for the boundary

    Returns:
        Boundary object
    """
    if boundary_type == "square":
        return SquareBoundary(size)
    elif boundary_type == "triangle":
        return TriangleBoundary(size)
    elif boundary_type == "circle":
        return CircleBoundary(size)
    else:
        raise ValueError(f"Unknown boundary type: {boundary_type}")
