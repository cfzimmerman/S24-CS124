use std::{collections::HashMap, fmt::Debug, hash::Hash, ops::Index};

pub trait Decr<K> {
    /// Decreases self by amount
    fn decr(&mut self, amt: K);
}

impl Decr<isize> for isize {
    fn decr(&mut self, amt: isize) {
        *self -= amt;
    }
}

/// A binary min heap specifically designed for use with
/// Prim's algorithm or Dijkstra's algorithm.
///
/// This heap stores underlying data in a vector and tracks the index
/// of every element in separate HashMap. This enables O(1) lookup
/// and O(log n) key decr.
///
/// For safety, both the underlying vector and the hashmap must own keys, so
/// keys are cloned frequently. For this reason, this data structure
/// should only be used with data that implements Copy or is wrapped
/// in a smart pointer like `Rc<RefCell>`.
///
/// If using an `Rc<RefCell>`, DO NOT modify the internal state of the
/// value while it is in the heap. Doing so will break the heap invariant
/// and make further use of the heap pointless.
#[derive(Debug)]
pub struct PrimHeap<T> {
    tvec: TwoWayVec<T>,
}

impl<T> PrimHeap<T>
where
    T: PartialEq + Eq + PartialOrd + Hash + Clone + Debug,
{
    pub fn new() -> PrimHeap<T> {
        PrimHeap {
            tvec: TwoWayVec::new(),
        }
    }

    /// Creates a new heap from lst in O(n) time
    pub fn heapify(lst: Vec<T>) -> PrimHeap<T> {
        let mut heap = PrimHeap {
            tvec: TwoWayVec::init(lst),
        };
        for ind in (0..=(heap.tvec.len() / 2)).rev() {
            heap.bubble_down(ind);
        }
        heap
    }

    /// Returns the top element of the heap.
    pub fn peek(&self) -> Option<&T> {
        self.tvec.get(0)
    }

    /// Removes and returns the top element of the heap. None if the
    /// heap is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.tvec.is_empty() {
            return None;
        }
        let last_ind = self.tvec.len() - 1;
        self.tvec.swap(0, last_ind);
        let popped = self.tvec.pop();
        if !self.tvec.is_empty() {
            self.bubble_down(0);
        }
        return popped;
    }

    /// Adds a new value to the heap
    pub fn push(&mut self, val: T) {
        self.tvec.push(val);
        self.bubble_up(self.tvec.len() - 1);
    }

    pub fn len(&self) -> usize {
        self.tvec.len()
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
        let curr = &self.tvec.index(curr_ind);
        let parent_ind = Self::parent_ind(curr_ind);
        let parent = &self.tvec.index(parent_ind);
        if curr >= parent {
            return;
        }
        self.tvec.swap(curr_ind, parent_ind);
        return self.bubble_up(parent_ind);
    }

    /// Performs down-heapify on a value, respecting the min heap property
    fn bubble_down(&mut self, curr_ind: usize) {
        let curr = self.tvec.index(curr_ind);
        let l_ind = Self::left_child_ind(curr_ind);
        let r_ind = Self::right_child_ind(curr_ind);
        let (min_child, min_child_ind) = match (self.tvec.get(l_ind), self.tvec.get(r_ind)) {
            (Some(l), Some(r)) if l <= r => (l, l_ind),
            (Some(_), Some(r)) => (r, r_ind),
            (Some(l), None) => (l, l_ind),
            (None, Some(r)) => (r, r_ind),
            (None, None) => return,
        };
        if curr <= min_child {
            return;
        }
        self.tvec.swap(curr_ind, min_child_ind);
        return self.bubble_down(min_child_ind);
    }
}

type IndMap<T> = HashMap<T, usize>;

/// Wraps a vector and hashmap so that values can be looked up in O(1) time
/// both by index and by value. This implies all values in the underlying vector
/// are unique! This data type has the indexing properties of a vector but
/// the uniqueness properties of a Set.
///
/// It also shares the same warnings about cloning and interior mutablity as PrimHeap.
///
/// Unless stated otherwise, assume every method does the same as its
/// counterpart from std::Vec except mutations are also reflected in the
/// hashmap.
#[derive(Debug)]
struct TwoWayVec<T> {
    lst: Vec<T>,
    hmap: IndMap<T>,
}

// There are lots of assertions and `expect` lines in this data structure, but
// all are to maintained externally guaranteed invariants. If something panics,
// that's a fundamental error beneath the API.
impl<T> TwoWayVec<T>
where
    T: PartialEq + Eq + Hash + Clone,
{
    /// Instantiates both an empty tracked array and its tracker map.
    pub fn new() -> TwoWayVec<T> {
        TwoWayVec {
            lst: Vec::new(),
            hmap: HashMap::new(),
        }
    }

    /// Builds a new HashTrackedArr from an existing Vector.
    /// If values in the input list are duplicate, only the first value
    /// is added to the returned data structure. Others are discarded.
    pub fn init(lst: Vec<T>) -> Self {
        let mut rvec = TwoWayVec::new();
        rvec.lst.reserve(lst.len());
        rvec.hmap.reserve(lst.len());
        for el in lst {
            rvec.push(el);
        }
        rvec
    }

    pub fn len(&self) -> usize {
        self.lst.len()
    }

    pub fn get(&self, ind: usize) -> Option<&T> {
        self.lst.get(ind)
    }

    pub fn index(&self, ind: usize) -> &T {
        self.lst.index(ind)
    }

    pub fn is_empty(&self) -> bool {
        self.lst.is_empty()
    }

    /// Not a vector method:
    /// Uses the hashmap to retrieve the index of a key. Returns
    /// None if the key could not be found.
    pub fn get_index(&self, el: &T) -> Option<&usize> {
        self.hmap.get(el)
    }

    /// Pushing a value already in the vec will yield None, and
    /// the redundant value will not be inserted.
    pub fn push(&mut self, el: T) -> Option<()> {
        if self.hmap.contains_key(&el) {
            return None;
        }
        self.lst.push(el.clone());
        self.hmap.insert(el, self.lst.len() - 1);
        Some(())
    }

    pub fn pop(&mut self) -> Option<T> {
        let popped = self.lst.pop();
        if let Some(el) = &popped {
            let popped_ind = self
                .hmap
                .remove(el)
                .expect("Index of el should have been tracked");
            assert_eq!(
                popped_ind,
                self.lst.len(),
                "Popped index should have matched map"
            );
        }
        popped
    }

    pub fn swap(&mut self, ind1: usize, ind2: usize) {
        self.lst.swap(ind1, ind2);
        for ind in [ind1, ind2] {
            let prev = self
                .hmap
                .get_mut(&self.lst[ind])
                .expect("Index of prev should have been tracked");
            *prev = ind;
        }
    }
}

#[cfg(test)]
mod heap_tests {
    use super::PrimHeap;
    use crate::{error::PsetRes, test_data::get_isize_arr};

    /// Asserts the index held in hmap is equal to the index
    /// of the actual element in the vector.
    fn assert_hmap_inds(heap: &PrimHeap<isize>) {
        assert_eq!(
            heap.tvec.lst.len(),
            heap.tvec.hmap.len(),
            "Vec and hmap tracker should be the same size."
        );
        for (ind, val) in heap.tvec.lst.iter().enumerate() {
            let in_hmap = heap.tvec.hmap.get(val).expect("Value should be in hmap");
            assert_eq!(
                &ind, in_hmap,
                "Element index should equal element value in hmap"
            );
        }
    }

    #[test]
    fn heapsort() -> PsetRes<()> {
        let mut nums = get_isize_arr()?;

        let mut insertion_heap = PrimHeap::new();
        for num in nums.iter().copied() {
            insertion_heap.push(num);
        }
        let mut heapify_heap = PrimHeap::heapify(nums.clone());

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
            let heapify_val = heapify_heap
                .pop()
                .expect("Heapify heap should have as many elements as array");
            let insertion_val = insertion_heap
                .pop()
                .expect("Insertion heap should have as many elements as array");
            assert_eq!(
                arr_val, heapify_val,
                "Nums and heapify heap should be sorted the same"
            );
            assert_eq!(
                arr_val, insertion_val,
                "Nums and insertion heap should be sorted the same"
            );

            assert_hmap_inds(&heapify_heap);
            assert_hmap_inds(&insertion_heap);
        }

        Ok(())
    }
}
