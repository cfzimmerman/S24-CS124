use std::{collections::HashMap, fmt::Debug, hash::Hash};

/// Tracks the weight of an entry in the heap. Because PrimHeap is a min heap,
/// the lightest elements float to the top.
#[derive(PartialEq, Eq, PartialOrd, Debug, Copy, Clone, Default)]
pub struct Weight<W>(W);

impl<W> Weight<W>
where
    W: Copy,
{
    pub fn new(val: W) -> Weight<W> {
        Weight(val)
    }

    pub fn get(&self) -> &W {
        &self.0
    }

    pub fn take(self) -> W {
        self.0
    }
}

impl From<f64> for Weight<f64> {
    fn from(value: f64) -> Self {
        Weight(value)
    }
}

/// A binary min heap specifically designed for use with
/// Prim's algorithm or Dijkstra's algorithm.
///
/// For safety, both the underlying vector and the hashmap must own keys, so
/// keys are cloned on insertion. For this reason, this data structure
/// should only be used with data that implements Copy or is wrapped
/// in a smart pointer like `Rc<T>`.
///
/// At the cost of requiring entry weights and relying on the added allocations
/// mentioned above, this heap offers the following characteristics:
/// - O(log n) upsert_min
/// - O(log n) take_min
/// - O(1) attribute retrieval
/// - O(1) peelkl
#[derive(Debug, Default)]
pub struct PrimHeap<T, W> {
    tvec: TwoWayVec<T, W>,
}

impl<T, W> PrimHeap<T, W>
where
    T: PartialEq + Eq + Hash + Clone + Debug + Default,
    W: PartialOrd + Debug + Default,
{
    #[cfg(test)]
    pub fn new() -> PrimHeap<T, W> {
        PrimHeap::default()
    }

    /// Creates a new heap from lst in O(n) time
    pub fn heapify(lst: Vec<(&T, Weight<W>)>) -> PrimHeap<T, W> {
        let mut heap = PrimHeap {
            tvec: TwoWayVec::init(lst),
        };
        for ind in (0..=(heap.tvec.len() / 2)).rev() {
            heap.bubble_down(ind);
        }
        heap
    }

    /// Returns the top element of the heap.
    #[cfg(test)]
    pub fn peek(&self) -> Option<&T> {
        self.tvec.get(0)
    }

    /// Removes and returns the top element of the heap. None if the
    /// heap is empty.
    pub fn take_min(&mut self) -> Option<(T, Weight<W>)> {
        if self.tvec.is_empty() {
            return None;
        }
        let last_ind = self.tvec.len() - 1;
        self.tvec.swap(0, last_ind);
        let popped = self.tvec.pop();
        if !self.tvec.is_empty() {
            self.bubble_down(0);
        }
        popped
    }

    /// Returns the weight of a value in the heap or None if the
    /// vertex isn't currently in the heap.
    pub fn get_weight(&self, val: &T) -> Option<&Weight<W>> {
        self.tvec.get_weight(val)
    }

    /// If a value is not yet in the heap, inserts it. If it is in the heap and the
    /// new weight is lighter, updates the weight of the value. If the new weight is
    /// heavier, does nothing.
    ///
    /// To avoid unnecessary allocations on valuers already in the heap, `val` is cloned
    /// internaly only if it needs to be added to the heap.
    pub fn upsert_min(&mut self, val: &T, weight: Weight<W>) {
        if let Some(ind) = self.tvec.upsert_min(val, weight) {
            self.bubble_up(ind);
        };
    }

    fn parent_ind(ind: usize) -> usize {
        (ind - 1) / 2
    }

    fn left_child_ind(ind: usize) -> usize {
        2 * ind + 1
    }

    fn right_child_ind(ind: usize) -> usize {
        2 * ind + 2
    }

    /// Performs up-heapify on a value, respecting the min heap property
    fn bubble_up(&mut self, curr_ind: usize) {
        if curr_ind == 0 {
            return;
        }
        let curr_weight = self
            .tvec
            .get_weight_of_ind(curr_ind)
            .expect("Curr ind should be well-defined in bubble up.");
        let parent_ind = Self::parent_ind(curr_ind);
        let parent_weight = self
            .tvec
            .get_weight_of_ind(parent_ind)
            .expect("Parent ind should be well-defined in bubble up.");
        if curr_weight >= parent_weight {
            return;
        }
        self.tvec.swap(curr_ind, parent_ind);
        self.bubble_up(parent_ind);
    }

    /// Performs down-heapify on a value, respecting the min heap property
    fn bubble_down(&mut self, curr_ind: usize) {
        let curr_weight = self
            .tvec
            .get_weight_of_ind(curr_ind)
            .expect("Curr ind should be well-defined in bubble down.");
        let l_ind = Self::left_child_ind(curr_ind);
        let r_ind = Self::right_child_ind(curr_ind);
        let (min_child_weight, min_child_ind) = match (
            self.tvec.get_weight_of_ind(l_ind),
            self.tvec.get_weight_of_ind(r_ind),
        ) {
            (Some(l), Some(r)) if l <= r => (l, l_ind),
            (Some(_), Some(r)) => (r, r_ind),
            (Some(l), None) => (l, l_ind),
            (None, Some(r)) => (r, r_ind),
            (None, None) => return,
        };
        if curr_weight <= min_child_weight {
            return;
        }
        self.tvec.swap(curr_ind, min_child_ind);
        self.bubble_down(min_child_ind);
    }
}

/// Wraps a vector and hashmap so that values can be looked up in O(1) time
/// both by index and by value. This implies all values in the underlying vector
/// are unique! This data type has the indexing properties of a vector but
/// the uniqueness properties of a Set.
///
/// Unless stated otherwise, assume every method does the same as its
/// counterpart from std::Vec except mutations are also reflected in the
/// hashmap.
///
/// While value weights are tracked, methods called on this data type often break
/// the heap invariant. It's the requirement of the caller to restore heap
/// properties.
#[derive(Debug, Default)]
struct TwoWayVec<T, W> {
    lst: Vec<T>,
    hmap: HashMap<T, VecMeta<W>>,
}

#[derive(Debug)]
struct VecMeta<W> {
    ind: usize,
    weight: Weight<W>,
}

// There are lots of assertions and `expect` lines in this data structure, but
// all are to guarantee invariants. If something panics,
// that's a fundamental error beneath the API.
impl<T, W> TwoWayVec<T, W>
where
    T: PartialEq + Eq + Hash + Clone,
    W: PartialOrd + Debug,
{
    /// Instantiates both an empty tracked array and its tracker map.
    pub fn new() -> TwoWayVec<T, W> {
        TwoWayVec {
            lst: Vec::new(),
            hmap: HashMap::new(),
        }
    }

    /// Builds a new TwoWayVec from an existing Vector.
    /// If input values are duplicate, the lowest weight is chosen.
    pub fn init(lst: Vec<(&T, Weight<W>)>) -> Self {
        let mut rvec = TwoWayVec::new();
        rvec.lst.reserve(lst.len());
        rvec.hmap.reserve(lst.len());
        for (el, weight) in lst {
            rvec.upsert_min(el, weight);
        }
        rvec
    }

    pub fn len(&self) -> usize {
        debug_assert!(self.lst.len() == self.hmap.len());
        self.lst.len()
    }

    #[cfg(test)]
    pub fn get(&self, ind: usize) -> Option<&T> {
        self.lst.get(ind)
    }

    pub fn is_empty(&self) -> bool {
        self.lst.is_empty()
    }

    /*
    /// Not a vector method:
    /// Uses the hashmap to retrieve the index of a key. Returns
    /// None if the key could not be found.
    pub fn get_index(&self, el: &T) -> Option<&usize> {
        self.hmap.get(el).map(|entry| &entry.ind)
    }
    */

    /// Uses the hashmap to retrieve the weight of a key. Returns
    /// None if the key could not be found.
    pub fn get_weight(&self, el: &T) -> Option<&Weight<W>> {
        self.hmap.get(el).map(|entry| &entry.weight)
    }

    /// Uses the hashmap to retrieve the weight of the key stored
    /// at the given index. Returns None if the given index is
    /// out of bounds.
    pub fn get_weight_of_ind(&self, ind: usize) -> Option<&Weight<W>> {
        let Some(el) = self.lst.get(ind) else {
            return None;
        };
        let meta = &self
            .hmap
            .get(el)
            .expect("Every element in the array should be tracked in the hmap.");
        Some(&meta.weight)
    }

    /// (update/insert) Adds a value with weight to the vec. If the same value with lesser weight
    /// already exists, the value is overwritten with the lighter weight. The index
    /// of the upserted value is returned.
    /// `el` is cloned twice if the value is not yet tracked.
    pub fn upsert_min(&mut self, el: &T, weight: Weight<W>) -> Option<usize> {
        if let Some(entry) = self.hmap.get_mut(el) {
            if entry.weight <= weight {
                return None;
            }
            entry.weight = weight;
            return Some(entry.ind);
        }
        self.lst.push(el.clone());
        let new_meta = VecMeta {
            ind: self.lst.len() - 1,
            weight,
        };
        self.hmap.insert(el.clone(), new_meta);
        Some(self.lst.len() - 1)
    }

    pub fn pop(&mut self) -> Option<(T, Weight<W>)> {
        let popped = match self.lst.pop() {
            None => return None,
            Some(el) => el,
        };
        let popped_meta = self
            .hmap
            .remove(&popped)
            .expect("Index of el should have been tracked");
        debug_assert_eq!(
            popped_meta.ind,
            self.lst.len(),
            "Popped index should have matched map"
        );
        Some((popped, popped_meta.weight))
    }

    pub fn swap(&mut self, ind1: usize, ind2: usize) {
        self.lst.swap(ind1, ind2);
        for ind in [ind1, ind2] {
            let prev = self
                .hmap
                .get_mut(&self.lst[ind])
                .expect("Index of prev should have been tracked");
            prev.ind = ind;
        }
    }
}

#[cfg(test)]
mod heap_tests {
    use std::collections::HashSet;

    use super::{PrimHeap, Weight};
    use crate::{
        error::PsetRes,
        test_data::{get_isize_arr, get_weighted_nums},
    };

    /// Weights integer heap entries by the entry's value itself. This equates to sorting
    /// a collection of unique integers.
    #[test]
    fn heapsort_by_val() -> PsetRes<()> {
        let mut nums = get_isize_arr()?;

        // Build one heap by repeated insertion
        let mut insertion_heap = PrimHeap::new();
        for num in nums.iter() {
            insertion_heap.upsert_min(num, Weight(*num));
        }
        // Build another heap by heapify
        let mut heapify_heap =
            PrimHeap::heapify(nums.iter().map(|num| (num, Weight(*num))).collect());

        // Process the input integers so they match what we expect from the heaps
        nums.sort();
        let mut unique_nums: Vec<isize> = Vec::with_capacity(nums.len());
        for num in nums {
            if let Some(prev) = unique_nums.last() {
                if prev == &num {
                    continue;
                }
            }
            unique_nums.push(num);
        }

        for arr_val in unique_nums {
            // Verify correctness of peek. Implicitly tests heapify/upsert too because
            // an incorrect ordering will cause these to fail.
            assert_eq!(Some(&arr_val), heapify_heap.peek());
            assert_eq!(Some(&arr_val), insertion_heap.peek());

            // Verify correctness of take.
            {
                let heapify_val = heapify_heap
                    .take_min()
                    .expect("Heapify heap should have as many elements as array");
                let insertion_val = insertion_heap
                    .take_min()
                    .expect("Insertion heap should have as many elements as array");
                assert_eq!(
                    arr_val, heapify_val.0,
                    "Nums and heapify heap should be sorted the same"
                );
                assert_eq!(
                    arr_val, insertion_val.0,
                    "Nums and insertion heap should be sorted the same"
                );
            }

            // Verify the metadata trackers for both heaps are always in a valid state.
            for heap in [&heapify_heap, &insertion_heap] {
                assert_eq!(
                    heap.tvec.lst.len(),
                    heap.tvec.hmap.len(),
                    "Vec and hmap tracker should be the same size."
                );
                for (ind, val) in heap.tvec.lst.iter().enumerate() {
                    let in_hmap = heap.tvec.hmap.get(val).expect("Value should be in hmap");
                    assert_eq!(
                        &ind, &in_hmap.ind,
                        "Element index should equal element value in hmap"
                    );
                    assert_eq!(val, &in_hmap.weight.0, "Weight shouldn't have been mutated");
                }
            }
        }

        assert_eq!(heapify_heap.take_min(), None);
        assert_eq!(insertion_heap.take_min(), None);
        Ok(())
    }

    /// Builds test data from inputs with distinct f64 weights. Sorts input
    /// data based on weight and ensures heaps yield minimum-weight unique
    /// keys in ascending order.
    #[test]
    fn heapsort_by_weight() -> PsetRes<()> {
        let mut weighted_nums = get_weighted_nums()?;

        // Build one heap by repeated insertion
        let mut insertion_heap = PrimHeap::new();
        for (num, weight) in weighted_nums.iter() {
            insertion_heap.upsert_min(num, *weight);
        }
        // Build another heap by heapify
        let mut heapify_heap = PrimHeap::heapify(
            weighted_nums
                .iter()
                .map(|(num, weight)| (num, *weight))
                .collect(),
        );

        weighted_nums.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut vals_seen: HashSet<i64> = HashSet::new();

        for (num, _) in weighted_nums {
            if vals_seen.contains(&num) {
                continue;
            }
            // Verify correctness of peek. Implicitly tests heapify/upsert too because
            // an incorrect ordering will cause these to fail.
            assert_eq!(Some(&num), heapify_heap.peek(), "heapify_heap");
            assert_eq!(Some(&num), insertion_heap.peek(), "insertion_heap");

            // Verify correctness of take.
            {
                let heapify_val = heapify_heap
                    .take_min()
                    .expect("Heapify heap should have as many elements as array");
                let insertion_val = insertion_heap
                    .take_min()
                    .expect("Insertion heap should have as many elements as array");
                assert_eq!(
                    num, heapify_val.0,
                    "Nums and heapify heap should be sorted the same"
                );
                assert_eq!(
                    num, insertion_val.0,
                    "Nums and insertion heap should be sorted the same"
                );
            }

            vals_seen.insert(num);
        }

        assert_eq!(heapify_heap.take_min(), None);
        assert_eq!(insertion_heap.take_min(), None);
        Ok(())
    }
}
