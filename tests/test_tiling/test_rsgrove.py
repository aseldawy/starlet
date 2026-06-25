"""Unit tests for RSGrove spatial partitioner.

Tests cover:
- EnvelopeNDLite operations (merge, area, margin)
- Spatial partitioning with different configurations
- Weighted partitioning
- Edge cases (empty inputs, single points)
- R*-tree style split selection
"""
import numpy as np
import pytest

from starlet._internal.tiling.RSGrove import (
    EnvelopeNDLite,
    KDPartitionTree,
    RSGrovePartitioner,
    partition_points,
    partition_weighted_points,
)


class TestEnvelopeNDLite:
    """Test the N-dimensional envelope (bounding box) class."""

    def test_create_2d_envelope(self):
        """Test creating a 2D envelope from min/max arrays."""
        env = EnvelopeNDLite(np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        assert env.getCoordinateDimension() == 2
        assert env.getMinCoord(0) == 0.0
        assert env.getMaxCoord(0) == 10.0

    def test_envelope_normalization(self):
        """Test that envelopes auto-normalize swapped min/max."""
        env = EnvelopeNDLite(np.array([10.0, 10.0]), np.array([0.0, 0.0]))
        assert env.getMinCoord(0) == 0.0
        assert env.getMaxCoord(0) == 10.0

    def test_empty_envelope(self):
        """Test detection of empty envelopes."""
        env = EnvelopeNDLite(np.array([0.0]), np.array([0.0]))
        env.setEmpty()  # Use setEmpty() method
        assert env.isEmpty()

    def test_merge_point(self):
        """Test expanding envelope to include a point."""
        env = EnvelopeNDLite(np.array([0.0, 0.0]), np.array([5.0, 5.0]))
        env.merge_point(np.array([10.0, 3.0]))
        assert env.getMaxCoord(0) == 10.0
        assert env.getMaxCoord(1) == 5.0

    def test_merge_box(self):
        """Test merging two envelopes."""
        env1 = EnvelopeNDLite(np.array([0.0, 0.0]), np.array([5.0, 5.0]))
        env2 = EnvelopeNDLite(np.array([3.0, 3.0]), np.array([10.0, 10.0]))
        env1.merge_box(env2)
        assert env1.getMinCoord(0) == 0.0
        assert env1.getMaxCoord(0) == 10.0

    def test_area_2d(self):
        """Test area calculation for 2D envelope."""
        env = EnvelopeNDLite(np.array([0.0, 0.0]), np.array([10.0, 5.0]))
        assert env.area() == 50.0

    def test_margin_2d(self):
        """Test margin (perimeter proxy) calculation."""
        env = EnvelopeNDLite(np.array([0.0, 0.0]), np.array([10.0, 5.0]))
        # Margin is sum of side lengths
        assert env.margin() == 15.0

    def test_from_points(self):
        """Test creating envelope from point array."""
        coords = np.array([[0.0, 5.0, 10.0], [0.0, 3.0, 7.0]])
        env = EnvelopeNDLite.from_points(coords)
        assert env.getMinCoord(0) == 0.0
        assert env.getMaxCoord(0) == 10.0
        assert env.getMinCoord(1) == 0.0
        assert env.getMaxCoord(1) == 7.0


class TestKDPartitionTree:
    """Test compact kd-tree partition lookup helper."""

    def test_single_leaf_finalize_searches_to_only_partition(self):
        tree = KDPartitionTree()

        assert tree.assign_partition_ids() == 1
        assert tree.rows.shape == (0, 4)
        assert tree.search(100.0, -100.0) == 0

    def test_finalize_and_search(self):
        tree = KDPartitionTree()

        assert len(tree) == 1
        assert np.isnan(tree.rows[0, 0])
        assert tree.rows[0, 1:].tolist() == [-1.0, 0.0, 0.0]

        lower, higher = tree.add_split(0, 0, 5.0)
        assert (lower, higher) == (1, 2)

        lower_lower, lower_higher = tree.add_split(lower, 1, 3.0)
        assert (lower_lower, lower_higher) == (3, 4)

        assert tree.rows[:, 1:].tolist() == [
            [0.0, 1.0, 2.0],
            [1.0, 3.0, 4.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
        assert tree.rows[:2, 0].tolist() == [5.0, 3.0]
        assert np.isnan(tree.rows[2:, 0]).all()

        assert tree.assign_partition_ids() == 3
        assert tree.rows.tolist() == [
            [5.0, 0.0, 1.0, -3.0],
            [3.0, 1.0, -1.0, -2.0],
        ]

        assert tree.search(1.0, 1.0) == 0
        assert tree.search(1.0, 4.0) == 1
        assert tree.search(6.0, 1.0) == 2

        global_mbr = EnvelopeNDLite(np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        assert tree.getPartitionMBR(0, global_mbr).mins.tolist() == [0.0, 0.0]
        assert tree.getPartitionMBR(0, global_mbr).maxs.tolist() == [5.0, 3.0]
        assert tree.getPartitionMBR(1, global_mbr).mins.tolist() == [0.0, 3.0]
        assert tree.getPartitionMBR(1, global_mbr).maxs.tolist() == [5.0, 10.0]
        assert tree.getPartitionMBR(2, global_mbr).mins.tolist() == [5.0, 0.0]
        assert tree.getPartitionMBR(2, global_mbr).maxs.tolist() == [10.0, 10.0]


class TestPartitionFunctions:
    """Test the core partitioning algorithms."""

    def test_partition_points_simple(self):
        """Test partitioning a small set of points."""
        # Create 100 random points in 2D
        np.random.seed(42)
        coords = np.random.rand(2, 100) * 100.0

        boxes = partition_points(coords, min_cap=10, max_cap=30, fraction_min_split=0.0)

        assert len(boxes) > 0
        # With 100 points and max_cap=30, expect 4-10 partitions
        assert 3 <= len(boxes) <= 12

    def test_partition_weighted_points(self):
        """Test weighted partitioning."""
        np.random.seed(42)
        coords = np.random.rand(2, 50) * 100.0
        weights = np.random.randint(1, 10, size=50).astype(float)

        boxes = partition_weighted_points(
            coords, weights,
            min_cap_w=20.0, max_cap_w=60.0, fraction_min_split=0.0
        )

        assert len(boxes) > 0

    def test_partition_preserves_all_points(self):
        """Test that all points fall within some partition."""
        np.random.seed(42)
        coords = np.random.rand(2, 50) * 100.0

        partitioner = partition_points(coords, min_cap=5, max_cap=15, fraction_min_split=0.0)
        global_mbr = EnvelopeNDLite.from_points(coords)
        boxes = [
            partitioner.getPartitionMBR(partition_id, global_mbr)
            for partition_id in range(partitioner.numPartitions())
        ]

        # Check each point is covered by at least one box
        for i in range(coords.shape[1]):
            point = coords[:, i]
            covered = False
            for box in boxes:
                if np.all(box.mins <= point) and np.all(point <= box.maxs):
                    covered = True
                    break
            assert covered, f"Point {i} not covered by any partition"

    def test_partition_empty_input(self):
        """Test partitioning with no points."""
        coords = np.empty((2, 0), dtype=float)
        partitioner = partition_points(coords, min_cap=1, max_cap=10, fraction_min_split=0.0)
        # Should handle gracefully, may return empty or single box
        assert isinstance(partitioner, KDPartitionTree)


class TestRSGrovePartitioner:
    """Test the main RSGrove partitioner class."""

    def test_construct_unweighted(self, mock_summary, sample_coords_2d):
        """Test constructing partitions without histogram weighting."""
        partitioner = RSGrovePartitioner()

        partitioner.construct(mock_summary, sample_coords_2d,
                             histogram=None, numPartitions=5)

        assert partitioner.numPartitions() > 0

    def test_construct_weighted(self, mock_summary, mock_histogram):
        """Test constructing partitions with histogram weighting."""
        partitioner = RSGrovePartitioner()

        # Sample some points
        np.random.seed(42)
        sample = np.random.rand(2, 50) * 100.0

        partitioner.construct(mock_summary, sample,
                             histogram=mock_histogram, numPartitions=3)

        assert partitioner.numPartitions() > 0


    def test_overlap_partition_single(self, mock_summary, sample_coords_2d):
        """Test selecting single best partition for an envelope."""
        partitioner = RSGrovePartitioner()
        partitioner.construct(mock_summary, sample_coords_2d,
                             histogram=None, numPartitions=5)

        query = EnvelopeNDLite(np.array([20.0, 20.0]), np.array([30.0, 30.0]))
        pid = partitioner.overlapPartition(query)

        assert 0 <= pid < partitioner.numPartitions()

    def test_empty_envelope_random_assignment(self, mock_summary, sample_coords_2d):
        """Test that empty envelopes get random partition assignment."""
        partitioner = RSGrovePartitioner()
        partitioner.construct(mock_summary, sample_coords_2d,
                             histogram=None, numPartitions=5)

        result = partitioner.overlapPartition([])

        # Should assign to exactly one random partition
        assert 0 <= result < partitioner.numPartitions()

    def test_compute_point_weights(self, mock_histogram):
        """Test weight computation from histogram."""
        np.random.seed(42)
        sample = np.random.rand(2, 30) * 100.0

        weights = RSGrovePartitioner.computePointWeights(sample, mock_histogram)

        assert len(weights) == 30
        assert np.all(weights >= 0)
        assert weights.dtype == np.int64

    def test_partition_with_fabricated_points(self, mock_summary):
        """Test that empty sample triggers point fabrication."""
        partitioner = RSGrovePartitioner()

        empty_sample = np.empty((2, 0), dtype=float)
        partitioner.construct(mock_summary, empty_sample,
                             histogram=None, numPartitions=3)

        # Should still create partitions using fabricated points
        assert partitioner.numPartitions() > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point_partition(self):
        """Test partitioning a single point."""
        coords = np.array([[5.0], [5.0]])
        partitioner = partition_points(coords, min_cap=1, max_cap=10, fraction_min_split=0.0)
        assert partitioner.numPartitions() == 1

    def test_two_points_partition(self):
        """Test partitioning exactly two points."""
        coords = np.array([[0.0, 10.0], [0.0, 10.0]])
        partitioner = partition_points(coords, min_cap=1, max_cap=1, fraction_min_split=0.0)
        assert partitioner.numPartitions() == 2

    def test_collinear_points(self):
        """Test partitioning collinear points."""
        coords = np.array([[float(i) for i in range(20)],
                          [0.0] * 20])
        partitioner = partition_points(coords, min_cap=2, max_cap=5, fraction_min_split=0.0)
        assert partitioner.numPartitions() >= 4
