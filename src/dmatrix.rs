use libc::{c_float, c_uint};
use std::ffi::CStr;
use std::{ffi, path::Path, ptr, slice};

use super::{XGBError, XGBResult};

static KEY_GROUP_PTR: &CStr = c"group_ptr";
static KEY_GROUP: &CStr = c"group";
static KEY_LABEL: &CStr = c"label";
static KEY_WEIGHT: &CStr = c"weight";
static KEY_BASE_MARGIN: &CStr = c"base_margin";

/// Creates a JSON-encoded array interface string for f32 data.
pub(crate) fn make_array_interface_f32(data: &[f32]) -> String {
    let ptr = data.as_ptr() as usize;
    let len = data.len();
    format!(
        r#"{{"data":[{},false],"shape":[{}],"strides":null,"typestr":"<f4","version":3}}"#,
        ptr, len
    )
}

/// Creates a JSON-encoded array interface string for u64 data.
///
/// This function is used for CSR/CSC indices and indptr arrays.
/// XGBoost's C API expects uint64_t (bst_ulong) for these arrays,
/// so we use fixed-width u64 to ensure cross-platform compatibility.
pub(crate) fn make_array_interface_u64(data: &[u64]) -> String {
    let ptr = data.as_ptr() as usize;
    let len = data.len();
    format!(
        r#"{{"data":[{},false],"shape":[{}],"strides":null,"typestr":"<u8","version":3}}"#,
        ptr, len
    )
}

/// Creates a JSON-encoded array interface string for u32 data.
fn make_array_interface_u32(data: &[u32]) -> String {
    let ptr = data.as_ptr() as usize;
    let len = data.len();
    format!(
        r#"{{"data":[{},false],"shape":[{}],"strides":null,"typestr":"<u4","version":3}}"#,
        ptr, len
    )
}

/// Payload for a single batch pushed into a proxy DMatrix: either a dense array
/// interface, or the three CSR array interfaces plus the column count.
enum BatchData {
    Dense(ffi::CString),
    Csr {
        indptr: ffi::CString,
        indices: ffi::CString,
        values: ffi::CString,
        num_cols: usize,
    },
}

/// Single-batch data iterator used to build a `QuantileDMatrix` from an
/// in-memory matrix via XGBoost's callback API. It yields the whole matrix as a
/// single batch, then signals end-of-iteration.
struct BatchIter {
    /// Batch payload (array interfaces referencing caller data).
    data: BatchData,
    /// Optional labels as an array interface; the referenced data must outlive
    /// construction. Prebuilt so the FFI callback below stays panic-free.
    labels: Option<ffi::CString>,
    /// Proxy DMatrix the callbacks push data into.
    proxy: xgboost_sys::DMatrixHandle,
    /// Whether the single batch has already been yielded.
    yielded: bool,
}

/// `next` callback: push the one batch into the proxy, or report end-of-iteration.
/// Returns 1 while a batch is available, 0 at the end, -1 on failure. Must not
/// unwind across the FFI boundary, so it avoids any panicking operations.
unsafe extern "C" fn batch_iter_next(handle: xgboost_sys::DataIterHandle) -> i32 {
    let it = unsafe { &mut *(handle as *mut BatchIter) };
    if it.yielded {
        return 0;
    }
    let ret = match &it.data {
        BatchData::Dense(interface) => unsafe { xgboost_sys::XGProxyDMatrixSetDataDense(it.proxy, interface.as_ptr()) },
        BatchData::Csr {
            indptr,
            indices,
            values,
            num_cols,
        } => unsafe {
            xgboost_sys::XGProxyDMatrixSetDataCSR(
                it.proxy,
                indptr.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
                *num_cols as xgboost_sys::bst_ulong,
            )
        },
    };
    if ret != 0 {
        return -1;
    }
    if let Some(interface) = &it.labels {
        let ret =
            unsafe { xgboost_sys::XGDMatrixSetInfoFromInterface(it.proxy, c"label".as_ptr(), interface.as_ptr()) };
        if ret != 0 {
            return -1;
        }
    }
    it.yielded = true;
    1
}

/// `reset` callback: rewind so the batch can be yielded again.
unsafe extern "C" fn batch_iter_reset(handle: xgboost_sys::DataIterHandle) {
    let it = unsafe { &mut *(handle as *mut BatchIter) };
    it.yielded = false;
}

/// Data matrix used throughout XGBoost for training/predicting [`Booster`](struct.Booster.html) models.
///
/// It's used as a container for both features (i.e. a row for every instance), and an optional true label for that
/// instance (as an `f32` value).
///
/// Can be created files, or from dense or sparse
/// ([CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
/// or [CSC](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS))) matrices.
///
/// # Examples
///
/// ## Load from file
///
/// Load matrix from file in [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) or binary format.
///
/// ```should_panic
/// use xgb::DMatrix;
///
/// let dmat = DMatrix::load(r#"{"uri": "somefile.txt?format=csv"}"#).unwrap();
/// ```
///
/// ## Create from dense array
///
/// ```
/// use xgb::DMatrix;
///
/// let data = &[1.0, 0.5, 0.2, 0.2,
///              0.7, 1.0, 0.1, 0.1,
///              0.2, 0.0, 0.0, 1.0];
/// let num_rows = 3;
/// let mut dmat = DMatrix::from_dense(data, num_rows).unwrap();
/// assert_eq!(dmat.shape(), (3, 4));
///
/// // set true labels for each row
/// dmat.set_labels(&[1.0, 0.0, 1.0]);
/// ```
///
/// ## Create from sparse CSR matrix
///
/// Create from sparse representation of
/// ```text
/// [[1.0, 0.0, 2.0],
///  [0.0, 0.0, 3.0],
///  [4.0, 5.0, 6.0]]
/// ```
///
/// ```
/// use xgb::DMatrix;
///
/// let indptr: &[u64] = &[0, 1, 2, 6];
/// let indices: &[u64] = &[0, 2, 2, 0, 1, 2];
/// let data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let dmat = DMatrix::from_csc(indptr, indices, data, None).unwrap();
/// assert_eq!(dmat.shape(), (3, 3));
/// ```
#[derive(Debug)]
pub struct DMatrix {
    pub(super) handle: xgboost_sys::DMatrixHandle,
    num_rows: usize,
    num_cols: usize,
}

impl DMatrix {
    /// Construct a new instance from a DMatrixHandle created by the XGBoost C API.
    fn new(handle: xgboost_sys::DMatrixHandle) -> XGBResult<Self> {
        // number of rows/cols are frequently read throughout applications, so more convenient to pull them out once
        // when the matrix is created, instead of having to check errors each time XGDMatrixNum* is called
        let mut out = 0;
        xgb_call!(xgboost_sys::XGDMatrixNumRow(handle, &mut out))?;
        let num_rows = out as usize;

        let mut out = 0;
        xgb_call!(xgboost_sys::XGDMatrixNumCol(handle, &mut out))?;
        let num_cols = out as usize;

        trace!("Loaded DMatrix with shape: {}x{}", num_rows, num_cols);
        Ok(DMatrix {
            handle,
            num_rows,
            num_cols,
        })
    }

    /// Create a new `DMatrix` from dense array in row-major order.
    ///
    /// E.g. the matrix
    /// ```text
    /// [[1.0, 2.0],
    ///  [3.0, 4.0],
    ///  [5.0, 6.0]]
    /// ```
    /// would be represented converted into a `DMatrix` with
    /// ```
    /// use xgb::DMatrix;
    ///
    /// let data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let num_rows = 3;
    /// let dmat = DMatrix::from_dense(data, num_rows).unwrap();
    /// ```
    pub fn from_dense(data: &[f32], num_rows: usize) -> XGBResult<Self> {
        // Three-way benchmark of the dense creation APIs
        // (benches/dmatrix_benchmark.rs, `from_dense` group):
        //   - XGDMatrixCreateFromMat: fastest below ~50k elements (no thread setup)
        //   - XGDMatrixCreateFromMat_omp: fastest above (~1.5x the array-interface
        //     XGDMatrixCreateFromDense at 500k+, ~5x plain CreateFromMat)
        // Neither is deprecated, so dispatch on the measured crossover. NaN is the
        // missing value in both paths, matching historical behavior.
        const PARALLEL_THRESHOLD: usize = 50_000;

        let mut handle = ptr::null_mut();
        let num_cols = (data.len() / num_rows) as xgboost_sys::bst_ulong;

        if data.len() < PARALLEL_THRESHOLD {
            xgb_call!(xgboost_sys::XGDMatrixCreateFromMat(
                data.as_ptr(),
                num_rows as xgboost_sys::bst_ulong,
                num_cols,
                f32::NAN,
                &mut handle
            ))?;
        } else {
            xgb_call!(xgboost_sys::XGDMatrixCreateFromMat_omp(
                data.as_ptr(),
                num_rows as xgboost_sys::bst_ulong,
                num_cols,
                f32::NAN,
                &mut handle,
                0 // <=0 means use all available cores
            ))?;
        }
        DMatrix::new(handle)
    }

    /// Create a `QuantileDMatrix` from a dense row-major matrix.
    ///
    /// A `QuantileDMatrix` stores the data pre-binned for the `hist` tree method
    /// (the XGBoost default), using far less memory than a regular `DMatrix`
    /// (~1 byte per feature value instead of 4) and skipping a separate sketching
    /// pass during training. Prefer it for large in-memory training sets.
    ///
    /// `max_bin` must match the booster's `max_bin` training parameter (default
    /// 256), otherwise training errors out. Labels, when given, are attached
    /// during construction (the supported path for quantile matrices) and must
    /// have length `num_rows`.
    ///
    /// Note: a `QuantileDMatrix` is intended for training with `hist`; unlike a
    /// regular `DMatrix` it cannot be serialised with [`save`](Self::save).
    pub fn from_dense_quantile(data: &[f32], num_rows: usize, labels: Option<&[f32]>, max_bin: u32) -> XGBResult<Self> {
        if let Some(l) = labels {
            assert_eq!(l.len(), num_rows, "labels length must equal num_rows");
        }
        let num_cols = data.len() / num_rows;
        let ptr = data.as_ptr() as usize;
        let data_interface = format!(
            r#"{{"data":[{},false],"shape":[{},{}],"strides":null,"typestr":"<f4","version":3}}"#,
            ptr, num_rows, num_cols
        );
        let data_cstr = ffi::CString::new(data_interface).unwrap();

        Self::quantile_from_batch(BatchData::Dense(data_cstr), labels, max_bin)
    }

    /// Create a `QuantileDMatrix` from a sparse CSR matrix, the sparse
    /// counterpart of [`from_dense_quantile`](Self::from_dense_quantile).
    ///
    /// Same CSR layout as [`from_csr`](Self::from_csr); `num_cols` is required
    /// (the proxy DMatrix cannot infer it). See `from_dense_quantile` for the
    /// `max_bin`/labels semantics and usage caveats.
    pub fn from_csr_quantile(
        indptr: &[u64],
        indices: &[u64],
        data: &[f32],
        num_cols: usize,
        labels: Option<&[f32]>,
        max_bin: u32,
    ) -> XGBResult<Self> {
        assert_eq!(indices.len(), data.len());
        if let Some(l) = labels {
            assert_eq!(l.len(), indptr.len() - 1, "labels length must equal num_rows");
        }
        let batch = BatchData::Csr {
            indptr: ffi::CString::new(make_array_interface_u64(indptr)).unwrap(),
            indices: ffi::CString::new(make_array_interface_u64(indices)).unwrap(),
            values: ffi::CString::new(make_array_interface_f32(data)).unwrap(),
            num_cols,
        };
        Self::quantile_from_batch(batch, labels, max_bin)
    }

    /// Shared `QuantileDMatrix` construction from a single prepared batch.
    fn quantile_from_batch(data: BatchData, labels: Option<&[f32]>, max_bin: u32) -> XGBResult<Self> {
        let mut proxy = ptr::null_mut();
        xgb_call!(xgboost_sys::XGProxyDMatrixCreate(&mut proxy))?;

        let mut iter = BatchIter {
            data,
            labels: labels.map(|l| ffi::CString::new(make_array_interface_f32(l)).unwrap()),
            proxy,
            yielded: false,
        };

        let config = format!(r#"{{"missing": NaN, "max_bin": {}}}"#, max_bin);
        let config_cstr = ffi::CString::new(config).unwrap();

        let mut out = ptr::null_mut();
        // SAFETY: `iter` outlives the (synchronous) call; the callbacks only ever
        // dereference the handle we pass and never unwind across the boundary.
        let ret = unsafe {
            xgboost_sys::XGQuantileDMatrixCreateFromCallback(
                &mut iter as *mut BatchIter as xgboost_sys::DataIterHandle,
                proxy,
                ptr::null_mut(), // no reference matrix for quantile alignment
                Some(batch_iter_reset),
                Some(batch_iter_next),
                config_cstr.as_ptr(),
                &mut out,
            )
        };

        // Free the proxy regardless of outcome; surface the create error first.
        let proxy_free = unsafe { xgboost_sys::XGDMatrixFree(proxy) };
        XGBError::check_return_value(ret)?;
        XGBError::check_return_value(proxy_free)?;
        DMatrix::new(out)
    }

    /// Create a new `DMatrix` from a sparse
    /// [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) matrix.
    ///
    /// Uses standard CSR representation where the column indices for row _i_ are stored in
    /// `indices[indptr[i]:indptr[i+1]]` and their corresponding values are stored in
    /// `data[indptr[i]:indptr[i+1]`.
    ///
    /// If `num_cols` is set to None, number of columns will be inferred from given data.
    ///
    /// Note: For small matrices (< 30000 non-zero elements), single-threaded creation is used
    /// to avoid thread synchronization overhead. For larger matrices, multi-threaded creation
    /// is used for better performance.
    pub fn from_csr(indptr: &[u64], indices: &[u64], data: &[f32], num_cols: Option<usize>) -> XGBResult<Self> {
        // Threshold below which single-threaded is faster due to thread overhead.
        // Benchmarking shows crossover point is around 30k non-zeros on typical hardware.
        const SINGLE_THREAD_THRESHOLD: usize = 30000;

        assert_eq!(indices.len(), data.len());
        let mut handle = ptr::null_mut();
        let num_cols = num_cols.unwrap_or(0) as xgboost_sys::bst_ulong;

        let indptr_interface = make_array_interface_u64(indptr);
        let indices_interface = make_array_interface_u64(indices);
        let data_interface = make_array_interface_f32(data);

        let indptr_cstr = ffi::CString::new(indptr_interface).unwrap();
        let indices_cstr = ffi::CString::new(indices_interface).unwrap();
        let data_cstr = ffi::CString::new(data_interface).unwrap();

        // Use single thread for small matrices to avoid thread synchronization overhead
        let config: &CStr = if data.len() < SINGLE_THREAD_THRESHOLD {
            cr#"{"missing": NaN, "nthread": 1}"#
        } else {
            cr#"{"missing": NaN}"#
        };

        xgb_call!(xgboost_sys::XGDMatrixCreateFromCSR(
            indptr_cstr.as_ptr(),
            indices_cstr.as_ptr(),
            data_cstr.as_ptr(),
            num_cols,
            config.as_ptr(),
            &mut handle
        ))?;
        DMatrix::new(handle)
    }

    /// Create a new `DMatrix` from a sparse
    /// [CSC](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS))) matrix.
    ///
    /// Uses standard CSC representation where the row indices for column _i_ are stored in
    /// `indices[indptr[i]:indptr[i+1]]` and their corresponding values are stored in
    /// `data[indptr[i]:indptr[i+1]`.
    ///
    /// If `num_rows` is set to None, number of rows will be inferred from given data.
    ///
    /// Note: For small matrices (< 30000 non-zero elements), single-threaded creation is used
    /// to avoid thread synchronization overhead. For larger matrices, multi-threaded creation
    /// is used for better performance.
    pub fn from_csc(indptr: &[u64], indices: &[u64], data: &[f32], num_rows: Option<usize>) -> XGBResult<Self> {
        // Threshold below which single-threaded is faster due to thread overhead.
        // Benchmarking shows crossover point is around 30k non-zeros on typical hardware.
        const SINGLE_THREAD_THRESHOLD: usize = 30000;

        assert_eq!(indices.len(), data.len());
        let mut handle = ptr::null_mut();
        let num_rows = num_rows.unwrap_or(0) as xgboost_sys::bst_ulong;

        let indptr_interface = make_array_interface_u64(indptr);
        let indices_interface = make_array_interface_u64(indices);
        let data_interface = make_array_interface_f32(data);

        let indptr_cstr = ffi::CString::new(indptr_interface).unwrap();
        let indices_cstr = ffi::CString::new(indices_interface).unwrap();
        let data_cstr = ffi::CString::new(data_interface).unwrap();

        // Use single thread for small matrices to avoid thread synchronization overhead
        let config: &CStr = if data.len() < SINGLE_THREAD_THRESHOLD {
            cr#"{"missing": NaN, "nthread": 1}"#
        } else {
            cr#"{"missing": NaN}"#
        };

        xgb_call!(xgboost_sys::XGDMatrixCreateFromCSC(
            indptr_cstr.as_ptr(),
            indices_cstr.as_ptr(),
            data_cstr.as_ptr(),
            num_rows,
            config.as_ptr(),
            &mut handle
        ))?;
        DMatrix::new(handle)
    }

    /// Create a new `DMatrix` from given file.
    ///
    /// Supports text files in [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) format, CSV,
    /// binary files written either by `save`, or from another XGBoost library.
    ///
    /// For more details on accepted formats, seem the
    /// [XGBoost input format](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html)
    /// documentation.
    ///
    /// # LIBSVM format
    ///
    /// Specified data in a sparse format as:
    /// ```text
    /// <label> <index>:<value> [<index>:<value> ...]
    /// ```
    ///
    /// E.g.
    /// ```text
    /// 0 1:1 9:0 11:0
    /// 1 9:1 11:0.375 15:1
    /// 0 1:0 8:0.22 11:1
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        debug!("Loading DMatrix from: {}", path.as_ref().display());
        let mut handle = ptr::null_mut();
        let fname = crate::path_to_c_str(path);
        xgb_call!(xgboost_sys::XGDMatrixCreateFromURI(fname.as_ptr(), &mut handle))?;
        DMatrix::new(handle)
    }

    /// Load a `DMatrix` from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if the path contains non-UTF8 characters.
    pub fn load_binary<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        debug!("Loading DMatrix from: {}", path.as_ref().display());
        let mut handle = ptr::null_mut();
        // Use XGDMatrixCreateFromURI with a JSON config specifying the URI
        // Binary format is auto-detected, no format parameter needed
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| XGBError::new("Path contains non-UTF8 characters"))?;
        // Escape backslashes and quotes for valid JSON
        let escaped_path = path_str.replace('\\', "\\\\").replace('"', "\\\"");
        let config = format!(r#"{{"uri": "{}", "silent": 1}}"#, escaped_path);
        let config_cstr = ffi::CString::new(config).unwrap();
        xgb_call!(xgboost_sys::XGDMatrixCreateFromURI(config_cstr.as_ptr(), &mut handle))?;
        DMatrix::new(handle)
    }

    /// Serialise this `DMatrix` as a binary file to given path.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> XGBResult<()> {
        debug!("Writing DMatrix to: {}", path.as_ref().display());
        let fname: ffi::CString = crate::path_to_c_str(path);
        let silent = true;
        xgb_call!(xgboost_sys::XGDMatrixSaveBinary(
            self.handle,
            fname.as_ptr(),
            silent as i32
        ))
    }

    /// Get the number of rows in this matrix.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get the number of columns in this matrix.
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Get the shape (rows x columns) of this matrix.
    pub fn shape(&self) -> (usize, usize) {
        (self.num_rows(), self.num_cols())
    }

    /// Get a new DMatrix as a containing only given indices.
    pub fn slice(&self, indices: &[usize]) -> XGBResult<DMatrix> {
        debug!("Slicing {} rows from DMatrix", indices.len());
        let mut out_handle = ptr::null_mut();
        let indices: Vec<i32> = indices.iter().map(|x| *x as i32).collect();
        xgb_call!(xgboost_sys::XGDMatrixSliceDMatrix(
            self.handle,
            indices.as_ptr(),
            indices.len() as xgboost_sys::bst_ulong,
            &mut out_handle
        ))?;
        DMatrix::new(out_handle)
    }

    /// Get ground truth labels for each row of this matrix.
    pub fn get_labels(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_LABEL)
    }

    /// Set ground truth labels for each row of this matrix.
    pub fn set_labels(&mut self, array: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_LABEL, array)
    }

    /// Get weights of each instance.
    pub fn get_weights(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_WEIGHT)
    }

    /// Set weights of each instance.
    pub fn set_weights(&mut self, array: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_WEIGHT, array)
    }

    /// Get base margin.
    pub fn get_base_margin(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_BASE_MARGIN)
    }

    /// Set base margin.
    ///
    /// If specified, xgboost will start from this margin, can be used to specify initial prediction to boost from.
    pub fn set_base_margin(&mut self, array: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_BASE_MARGIN, array)
    }

    /// Set the index for the beginning and end of a group.
    ///
    /// Needed when the learning task is ranking.
    ///
    /// See the XGBoost documentation for more information.
    pub fn set_group(&mut self, group: &[u32]) -> XGBResult<()> {
        // same as xgb_call!(xgboost_sys::XGDMatrixSetGroup(self.handle, group.as_ptr(), group.len() as u64))
        self.set_uint_info(KEY_GROUP, group)
    }

    /// Get the index for the beginning and end of a group.
    ///
    /// Needed when the learning task is ranking.
    ///
    /// See the XGBoost documentation for more information.
    pub fn get_group(&self) -> XGBResult<&[u32]> {
        self.get_uint_info(KEY_GROUP_PTR)
    }

    fn get_float_info(&self, field: &CStr) -> XGBResult<&[f32]> {
        let mut out_len = 0;
        let mut out_dptr = ptr::null();
        xgb_call!(xgboost_sys::XGDMatrixGetFloatInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_dptr
        ))?;

        if out_len > 0 {
            Ok(unsafe { slice::from_raw_parts(out_dptr as *mut c_float, out_len as usize) })
        } else {
            Ok(&[0.0; 0])
        }
    }

    fn set_float_info(&mut self, field: &CStr, array: &[f32]) -> XGBResult<()> {
        xgb_call!(xgboost_sys::XGDMatrixSetFloatInfo(
            self.handle,
            field.as_ptr(),
            array.as_ptr(),
            array.len() as u64
        ))
    }

    fn get_uint_info(&self, field: &CStr) -> XGBResult<&[u32]> {
        let mut out_len = 0;
        let mut out_dptr = ptr::null();
        xgb_call!(xgboost_sys::XGDMatrixGetUIntInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_dptr
        ))?;

        if out_len > 0 {
            Ok(unsafe { slice::from_raw_parts(out_dptr as *mut c_uint, out_len as usize) })
        } else {
            Ok(&[0; 0])
        }
    }

    fn set_uint_info(&mut self, field: &CStr, array: &[u32]) -> XGBResult<()> {
        let array_interface = make_array_interface_u32(array);
        let data_cstr = ffi::CString::new(array_interface).unwrap();
        xgb_call!(xgboost_sys::XGDMatrixSetInfoFromInterface(
            self.handle,
            field.as_ptr(),
            data_cstr.as_ptr()
        ))
    }
}

impl Drop for DMatrix {
    fn drop(&mut self) {
        xgb_call!(xgboost_sys::XGDMatrixFree(self.handle)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read_train_matrix() -> XGBResult<DMatrix> {
        DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#)
    }

    #[test]
    fn read_matrix() {
        assert!(read_train_matrix().is_ok());
    }

    #[test]
    fn read_num_rows() {
        assert_eq!(read_train_matrix().unwrap().num_rows(), 6513);
    }

    #[test]
    fn read_num_cols() {
        assert_eq!(read_train_matrix().unwrap().num_cols(), 127);
    }

    #[test]
    fn writing_and_reading() {
        let dmat = read_train_matrix().unwrap();

        let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let out_path = tmp_dir.path().join("dmat.bin");
        dmat.save(&out_path).unwrap();

        let dmat2 = DMatrix::load_binary(out_path).unwrap();

        assert_eq!(dmat.num_rows(), dmat2.num_rows());
        assert_eq!(dmat.num_cols(), dmat2.num_cols());

        // Verify labels are preserved after save/load
        let labels1 = dmat.get_labels().unwrap();
        let labels2 = dmat2.get_labels().unwrap();
        assert_eq!(labels1.len(), labels2.len());
        assert_eq!(labels1, labels2);
    }

    #[test]
    fn get_set_labels() {
        let mut dmat = read_train_matrix().unwrap();
        let labels = dmat.get_labels();
        assert!(labels.is_ok());
        let mut labels = labels.unwrap().to_vec();
        assert_eq!(labels.len(), 6513);

        labels[0] = 0.1;
        assert_ne!(dmat.get_labels().unwrap(), labels);
        assert!(dmat.set_labels(&labels).is_ok());
        assert_eq!(dmat.get_labels().unwrap(), labels);
    }

    #[test]
    fn get_set_weights() {
        let mut dmat = read_train_matrix().unwrap();
        assert!(dmat.get_weights().unwrap().is_empty());

        let weight = [1.0, 10.0, 44.9555];
        assert!(dmat.set_weights(&weight).is_ok());
        assert_eq!(dmat.get_weights().unwrap(), weight);
    }

    #[test]
    fn get_set_base_margin() {
        let mut dmat = read_train_matrix().unwrap();
        let base_margin = dmat.get_base_margin();
        assert!(base_margin.is_ok());
        assert!(base_margin.unwrap().is_empty());

        let base_margin = vec![0.00001; dmat.num_rows()];
        assert!(dmat.set_base_margin(&base_margin).is_ok());
        assert_eq!(dmat.get_base_margin().unwrap(), base_margin);
    }

    #[test]
    fn get_set_group() {
        let mut dmat = read_train_matrix().unwrap();
        assert!(dmat.get_group().unwrap().is_empty());

        let group = [1];
        assert!(dmat.set_group(&group).is_ok());
        assert_eq!(dmat.get_group().unwrap(), &[0, 1]);
    }

    #[test]
    fn from_csr() {
        let indptr: [u64; 5] = [0, 2, 3, 6, 8];
        let indices: [u64; 8] = [0, 2, 2, 0, 1, 2, 1, 2];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dmat = DMatrix::from_csr(&indptr, &indices, &data, None).unwrap();
        assert_eq!(dmat.num_rows(), 4);
        assert_eq!(dmat.num_cols(), 0); // https://github.com/dmlc/xgboost/pull/7265

        let dmat = DMatrix::from_csr(&indptr, &indices, &data, Some(10)).unwrap();
        assert_eq!(dmat.num_rows(), 4);
        assert_eq!(dmat.num_cols(), 10);
    }

    #[test]
    fn from_csc() {
        let indptr: [u64; 5] = [0, 2, 3, 6, 8];
        let indices: [u64; 8] = [0, 2, 2, 0, 1, 2, 1, 2];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dmat = DMatrix::from_csc(&indptr, &indices, &data, None).unwrap();
        assert_eq!(dmat.num_rows(), 3);
        assert_eq!(dmat.num_cols(), 4);

        let dmat = DMatrix::from_csc(&indptr, &indices, &data, Some(10)).unwrap();
        assert_eq!(dmat.num_rows(), 10);
        assert_eq!(dmat.num_cols(), 4);
    }

    #[test]
    fn from_dense_quantile() {
        let num_rows = 50;
        let num_cols = 4;
        let mut data = vec![0f32; num_rows * num_cols];
        let mut labels = vec![0f32; num_rows];
        for i in 0..num_rows {
            for j in 0..num_cols {
                data[i * num_cols + j] = ((i * 3 + j) % 17) as f32;
            }
            labels[i] = (i % 2) as f32;
        }

        let qdm = DMatrix::from_dense_quantile(&data, num_rows, Some(&labels), 64).unwrap();
        assert_eq!(qdm.shape(), (num_rows, num_cols));
        assert_eq!(qdm.get_labels().unwrap(), &labels[..]);

        // Labels are optional.
        let qdm = DMatrix::from_dense_quantile(&data, num_rows, None, 64).unwrap();
        assert_eq!(qdm.shape(), (num_rows, num_cols));
    }

    #[test]
    fn from_dense() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_rows = 2;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.num_rows(), 2);
        assert_eq!(dmat.num_cols(), 3);

        let data = vec![1.0, 2.0, 3.0];
        let num_rows = 3;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.num_rows(), 3);
        assert_eq!(dmat.num_cols(), 1);
    }

    #[test]
    fn slice_from_indices() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let num_rows = 4;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.shape(), (4, 2));

        assert_eq!(dmat.slice(&[]).unwrap().shape(), (0, 2));
        assert_eq!(dmat.slice(&[1]).unwrap().shape(), (1, 2));
        assert_eq!(dmat.slice(&[0, 1]).unwrap().shape(), (2, 2));
        assert_eq!(dmat.slice(&[3, 2, 1]).unwrap().shape(), (3, 2));
        // slicing out of bounds is not safe and can cause a segfault
        // assert_eq!(dmat.slice(&[10, 11, 12]).unwrap().shape(), (3, 2));
    }

    #[test]
    fn slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let num_rows = 4;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.shape(), (4, 3));

        assert_eq!(dmat.slice(&[0, 1, 2, 3]).unwrap().shape(), (4, 3));
        assert_eq!(dmat.slice(&[0, 1]).unwrap().shape(), (2, 3));
        assert_eq!(dmat.slice(&[1, 0]).unwrap().shape(), (2, 3));
        assert_eq!(dmat.slice(&[0, 1, 2]).unwrap().shape(), (3, 3));
        assert_eq!(dmat.slice(&[3, 2, 1]).unwrap().shape(), (3, 3));
    }
}
