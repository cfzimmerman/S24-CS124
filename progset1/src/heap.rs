#[derive(Debug)]
pub struct MinHeap<T> {
    arr: Vec<T>,
}

impl<T> MinHeap<T>
where
    T: PartialEq + PartialOrd,
{
    pub fn new() -> MinHeap<T> {
        MinHeap { arr: vec![] }
    }

    /// Creates a new heap from lst in O(n) time
    pub fn heapify(lst: Vec<T>) -> MinHeap<T> {
        let mut heap = MinHeap { arr: lst };
        for ind in (0..=(heap.arr.len() / 2)).rev() {
            heap.bubble_down(ind);
        }
        heap
    }

    /// Returns the top element of the heap.
    pub fn peek(&self) -> Option<&T> {
        self.arr.get(0)
    }

    /// Removes and returns the top element of the heap. None if the
    /// heap is empty.
    pub fn pop(&mut self) -> Option<T> {
        match self.arr.len() {
            0 => return None,
            1 => return self.arr.pop(),
            _ => (),
        };
        let last_ind = self.arr.len() - 1;
        self.arr.swap(0, last_ind);
        let popped = self.arr.pop();
        self.bubble_down(0);
        return popped;
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
        let curr = &self.arr[curr_ind];
        let parent_ind = Self::parent_ind(curr_ind);
        let parent = match self.arr.get(parent_ind) {
            None => return,
            Some(p) => p,
        };
        if curr >= parent {
            return;
        }
        self.arr.swap(curr_ind, parent_ind);
        return self.bubble_up(parent_ind);
    }

    /// Performs down-heapify on a value, respecting the min heap property
    fn bubble_down(&mut self, curr_ind: usize) {
        let curr = &self.arr[curr_ind];
        let l_ind = Self::left_child_ind(curr_ind);
        let r_ind = Self::right_child_ind(curr_ind);
        let (min_child, min_child_ind) = match (self.arr.get(l_ind), self.arr.get(r_ind)) {
            (Some(l), Some(r)) if l <= r => (l, l_ind),
            (Some(_), Some(r)) => (r, r_ind),
            (Some(l), None) => (l, l_ind),
            (None, Some(r)) => (r, r_ind),
            (None, None) => return,
        };
        if curr <= min_child {
            return;
        }
        self.arr.swap(curr_ind, min_child_ind);
        return self.bubble_down(min_child_ind);
    }
}

#[cfg(test)]
mod heap_tests {
    use super::MinHeap;
    use crate::{error::PsetRes, test_data::get_isize_arr};

    #[test]
    fn heapsort() -> PsetRes<()> {
        let mut nums = get_isize_arr()?;
        let mut heap = MinHeap::heapify(nums.clone());
        nums.sort_by(|a, b| b.cmp(a));

        while let Some(arr_val) = nums.pop() {
            let heap_val = heap
                .pop()
                .expect("Heap should have as many elements as array");
            assert_eq!(heap_val, arr_val, "Heap and nums should be sorted the same");
        }

        Ok(())
    }
}
