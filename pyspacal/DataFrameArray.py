import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import h5py
import os
from matplotlib import pyplot as plt


class SeriesArray(pd.Series):
    @property
    def _constructor(self):
        return SeriesArray


class ShapeError(Exception):
    pass


class DataFrameArray(pd.DataFrame):
    """DataFrame with at least one array column

    standard DataFrames don't play well with array data in a single column. however, when dealing
    with timeseries or image data, it's common to have a small number of 1d or multi-dimensional
    data variables (e.g., the timeseries and a normalized versionof the timeseries) and many pieces
    of metadata that describe that data (e.g., subject id, trial number, etc.). in these
    situations, you typically don't want to grab specific time points; most of your operations will
    operating on the whole data, with functions that take the whole array as input. therefore, the
    standard DataFrame approach is unnecessary and unwieldy. DataFrameArray plays nicely with array
    data by storing them in a single column, allowing you to use metadata to select the specific
    cases you need.

    In order to access the data stored in the array columns, use .data.

    Example:

    test = DataFrameArray({'a': 'value', 'b': [1,2], 'c': [[1,2], [3,4]]})

    """

    _metadata = []

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        # find the shape of everything in the data dictionary. we assume that there are at most
        # three shapes in this set: empty shape (corresponding to one integer or string, the same
        # for each instance we construct), a 1d shape (either one value for each instance or, if
        # this is only one instance, the array-like data), and an nd shape (one >=1d array for each
        # instance). note that you must have the nd shape, the others are optional, and you can
        # have multiple objects with each shape. if you have nd and 1d, the first dimension of the
        # nd shape must be the same value as the 1d and the nd must be >=2d.
        if not isinstance(data, dict):
            super(DataFrameArray, self).__init__(data, index, columns, dtype, copy)
        else:
            data = data.copy()
            array_data = {}
            for k, v in data.items():
                if type(v) == ArrayData:
                    array_data[k] = v
            data.update(dict((k, v.data) for k, v in array_data.items()))
            shapes = set([np.array(v).shape for v in data.values()])
            try:
                shapes.remove(())
            except KeyError:
                warnings.warn("Assuming you have no dataframe-wide labels")
            if len(shapes) == 1:
                # in this case, we only have an nd shape, so we're only creating one instance,
                # i.e., one row in this DataFrame
                shapes = list(shapes)
                for k, v in data.items():
                    v = np.array(v)
                    if v.shape == shapes[0]:
                        array_data[k] = np.array(data.pop(k))
                        data[k] = ArrayData(v)
                if index is None:
                    index = [0]
            elif len(shapes) == 2:
                # in this case, we have both the nd and 1d shape, so we're creating multiple
                # instances, i.e., multiple rows in this DataFrame.  This ensures that shapes first
                # has the 1d and then the 2d shape
                shapes = sorted(list(shapes), key=lambda x: len(x))
                assert shapes[0][0] == shapes[1][0], "your nd data must have the same number of instances as your 1d!"
                for k, v in data.items():
                    v = np.array(v)
                    if v.shape == shapes[1]:
                        array_data[k] = np.array(data.pop(k))
                        data[k] = [ArrayData(v[i, :]) for i in range(shapes[1][0])]
            else:
                raise ShapeError("Data improperly shaped!")
            super(DataFrameArray, self).__init__(data, index, columns, dtype, copy)

    @property
    def _constructor(self):
        return DataFrameArray

    @property
    def _constructor_sliced(self):
        return SeriesArray

    def to_csv(self, path_or_buf=None, sep=",", na_rep='', float_format=None, columns=None,
               header=True, index=True, index_label=None, mode='w', encoding=None,
               compression='infer', quoting=None, quotechar='"', line_terminator=None,
               chunksize=None, tupleize_cols=None, date_format=None, doublequote=True,
               escapechar=None, decimal='.'):
        r"""Write object to a comma-separated values (csv) file and accompanying hdf5 file.

        This is just a wrapper around pandas.DataFrame.to_csv, except we will store any array data
        in an associated hdf5 file (it will have the same name, but different extension).

        .. versionchanged:: 0.24.0
            The order of arguments for Series was changed.

        Parameters
        ----------
        path_or_buf : str or file handle, default None
            File path or object, if None is provided the result is returned as
            a string.

            .. versionchanged:: 0.24.0

               Was previously named "path" for Series.

        sep : str, default ','
            String of length 1. Field delimiter for the output file.
        na_rep : str, default ''
            Missing data representation.
        float_format : str, default None
            Format string for floating point numbers.
        columns : sequence, optional
            Columns to write.
        header : bool or list of str, default True
            Write out the column names. If a list of strings is given it is
            assumed to be aliases for the column names.

            .. versionchanged:: 0.24.0

               Previously defaulted to False for Series.

        index : bool, default True
            Write row names (index).
        index_label : str or sequence, or False, default None
            Column label for index column(s) if desired. If None is given, and
            `header` and `index` are True, then the index names are used. A
            sequence should be given if the object uses MultiIndex. If
            False do not print fields for index names. Use index_label=False
            for easier importing in R.
        mode : str
            Python write mode, default 'w'.
        encoding : str, optional
            A string representing the encoding to use in the output file,
            defaults to 'ascii' on Python 2 and 'utf-8' on Python 3.
        compression : str, default 'infer'
            Compression mode among the following possible values: {'infer',
            'gzip', 'bz2', 'zip', 'xz', None}. If 'infer' and `path_or_buf`
            is path-like, then detect compression from the following
            extensions: '.gz', '.bz2', '.zip' or '.xz'. (otherwise no
            compression).

            .. versionchanged:: 0.24.0

               'infer' option added and set to default.

        quoting : optional constant from csv module
            Defaults to csv.QUOTE_MINIMAL. If you have set a `float_format`
            then floats are converted to strings and thus csv.QUOTE_NONNUMERIC
            will treat them as non-numeric.
        quotechar : str, default '\"'
            String of length 1. Character used to quote fields.
        line_terminator : string, optional
            The newline character or character sequence to use in the output
            file. Defaults to `os.linesep`, which depends on the OS in which
            this method is called ('\n' for linux, '\r\n' for Windows, i.e.).

            .. versionchanged:: 0.24.0
        chunksize : int or None
            Rows to write at a time.
        tupleize_cols : bool, default False
            Write MultiIndex columns as a list of tuples (if True) or in
            the new, expanded format, where each MultiIndex column is a row
            in the CSV (if False).

            .. deprecated:: 0.21.0
               This argument will be removed and will always write each row
               of the multi-index as a separate row in the CSV file.
        date_format : str, default None
            Format string for datetime objects.
        doublequote : bool, default True
            Control quoting of `quotechar` inside a field.
        escapechar : str, default None
            String of length 1. Character used to escape `sep` and `quotechar`
            when appropriate.
        decimal : str, default '.'
            Character recognized as decimal separator. E.g. use ',' for
            European data.

        Returns
        -------
        None or str
            If path_or_buf is None, returns the resulting csv format as a
            string. Otherwise returns None.

        See Also
        --------
        read_csv : Load a CSV file into a DataFrame.
        to_excel : Load an Excel file into a DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame({'name': ['Raphael', 'Donatello'],
        ...                    'mask': ['red', 'purple'],
        ...                    'weapon': ['sai', 'bo staff']})
        >>> df.to_csv(index=False)
        'name,mask,weapon\nRaphael,red,sai\nDonatello,purple,bo staff\n'

        """
        to_csv_args = locals().copy()
        to_csv_args.pop('self')
        to_csv_args.pop('__class__')
        if path_or_buf is None:
            raise NotImplementedError("DataFrameArray.to_csv requires a file to write to, so we "
                                      "can dump the ArrayData to an associated hdf5 file")
        else:
            hdf5_path = os.path.splitext(path_or_buf)[0] + ".hdf5"
        tmp = self.copy()
        with h5py.File(hdf5_path, 'w') as f:
            for i, row in tmp.iterrows():
                for col_name, val in row.iteritems():
                    if isinstance(val, ArrayData):
                        f.create_dataset("%s_%s" % (col_name, i), data=val.data)
                        tmp.loc[i, col_name] = "hdf5_%s_%s" % (col_name, i)
        super(DataFrameArray, tmp).to_csv(**to_csv_args)


class ArrayData(object):

    def __init__(self, value):
        """this is basically an array, except we pretend it's only 1d and change how it's printed, so
        DataFrames handle it well. in order to access actual array, use self.data basic arithmetic
        operators have been implemented, but you generally want to act on self.data

        """
        self.data = np.array(value)

    def __len__(self):
        return 1

    @property
    def shape(self):
        """note this is explicitly NOT what you want, use trueshape or ArrayData.data.shape instead
        """
        return ()

    @property
    def trueshape(self):
        """return the actual shape of the data
        """
        return self.data.shape

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        string_rep = "<%sd array of size %s>" % (self.data.ndim, list(self.data.shape))
        return string_rep

    def __getattr__(self, name):
        # this is necessary because we *do not* want ArrayData to be castable to an array in a
        # transparent manner; numpy.array should treat it as an object
        if name in ['__array_struct__', '__array_interface__', '__array__']:
            raise AttributeError("'ArrayData' object has no attribute '%s'" % name)
        return getattr(self.data, name)

    def __add__(self, other):
        try:
            res = self.data.__add__(other)
        except TypeError:
            # this happens when other is another ArrayData; in that case we need an explicit call
            # to .data because (in __getattr__) we made it impossible to transparently cast this as
            # an array
            res = self.data.__add__(other.data)
        return ArrayData(res)

    def __sub__(self, other):
        try:
            res = self.data.__sub__(other)
        except TypeError:
            # this happens when other is another ArrayData; in that case we need an explicit call
            # to .data because (in __getattr__) we made it impossible to transparently cast this as
            # an array
            res = self.data.__sub__(other.data)
        return ArrayData(res)

    def __mul__(self, other):
        try:
            res = self.data.__mul__(other)
        except TypeError:
            # this happens when other is another ArrayData; in that case we need an explicit call
            # to .data because (in __getattr__) we made it impossible to transparently cast this as
            # an array
            res = self.data.__mul__(other.data)
        return ArrayData(res)

    def __div__(self, other):
        try:
            res = self.data.__div__(other)
        except TypeError:
            # this happens when other is another ArrayData; in that case we need an explicit call
            # to .data because (in __getattr__) we made it impossible to transparently cast this as
            # an array
            res = self.data.__div__(other.data)
        return ArrayData(res)

    def __lt__(self, other):
        try:
            res = self.data.__lt__(other)
        except TypeError:
            # this happens when other is another ArrayData; in that case we need an explicit call
            # to .data because (in __getattr__) we made it impossible to transparently cast this as
            # an array
            res = self.data.__lt__(other.data)
        return res

    def __gt__(self, other):
        try:
            res = self.data.__gt__(other)
        except TypeError:
            # this happens when other is another ArrayData; in that case we need an explicit call
            # to .data because (in __getattr__) we made it impossible to transparently cast this as
            # an array
            res = self.data.__gt__(other.data)
        return res

    def __le__(self, other):
        try:
            res = self.data.__le__(other)
        except TypeError:
            # this happens when other is another ArrayData; in that case we need an explicit call
            # to .data because (in __getattr__) we made it impossible to transparently cast this as
            # an array
            res = self.data.__le__(other.data)
        return res

    def __ge__(self, other):
        try:
            res = self.data.__ge__(other)
        except TypeError:
            # this happens when other is another ArrayData; in that case we need an explicit call
            # to .data because (in __getattr__) we made it impossible to transparently cast this as
            # an array
            res = self.data.__ge__(other.data)
        return res

    def __eq__(self, other):
        try:
            res = self.data.__eq__(other)
        except TypeError:
            # this happens when other is another ArrayData; in that case we need an explicit call
            # to .data because (in __getattr__) we made it impossible to transparently cast this as
            # an array
            res = self.data.__eq__(other.data)
        return res

    def __ne__(self, other):
        try:
            res = self.data.__ne__(other)
        except TypeError:
            # this happens when other is another ArrayData; in that case we need an explicit call
            # to .data because (in __getattr__) we made it impossible to transparently cast this as
            # an array
            res = self.data.__ne__(other.data)
        return res


def set_ticklabels(datashape):
    xticklabels = datashape[0]/16
    if xticklabels == 0 or xticklabels == 1:
        xticklabels = True
    yticklabels = xticklabels
    return xticklabels, yticklabels


def ArrayData_heatmap(img, **kwargs):
    """plot heatmap using ArrayData

    this function is to be used with seaborn's FacetGrid (and FacetGrid.map), allowing the user to
    facet the DataFrameArray and plot the resulting data using sns.heatmap. any kwargs are passed
    directly to sns.heatmap

    img: ArrayData to be plotted. note that this implicitly assumes that img is 2d
    """
    if len(img) > 1:
        raise Exception("Didn't facet correctly, too many images meet this criteria! We can only "
                        "plot one")
    xticks, yticks = set_ticklabels(img.values[0].data.shape)
    sns.heatmap(img.values[0].data, xticklabels=xticks, yticklabels=yticks, **kwargs)


def ArrayData_imshow(img, cmap='gray', colorbar=False, **kwargs):
    """show image from ArrayData

    this function is to be used with seaborn's FacetGrid (and FacetGrid.map), allowing the user to
    facet the DataFrameArray and show the resulting image using plt.imshow any kwargs are passed
    directly to plt.imshow

    img: ArrayData to be plotted. note that this implicitly assumes that img is 2d
    """
    if len(img) > 1:
        raise Exception("Didn't facet correctly, too many images meet this criteria! We can only "
                        "plot one")
    kwargs.pop('color')
    fig = plt.gcf()
    ax = plt.gca()
    ax.set(xticks=[], yticks=[])
    ax.set_frame_on(False)
    mappable = ax.imshow(img.values[0].data, cmap=cmap, interpolation='none', **kwargs)
    if colorbar:
        fig.colorbar(mappable, ax=ax)


def read_csv(filepath_or_buffer, sep=',', delimiter=None, header='infer', names=None,
             index_col=None, usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True,
             dtype=None, engine=None, converters=None, true_values=None, false_values=None,
             skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None, na_values=None,
             keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True,
             parse_dates=False, infer_datetime_format=False, keep_date_col=False, date_parser=None,
             dayfirst=False, iterator=False, chunksize=None, compression='infer', thousands=None,
             decimal=b'.', lineterminator=None, quotechar='"', quoting=0, doublequote=True,
             escapechar=None, comment=None, encoding=None, dialect=None, tupleize_cols=None,
             error_bad_lines=True, warn_bad_lines=True, delim_whitespace=False, low_memory=True,
             memory_map=False, float_precision=None):
    """Read a comma-separated values (csv) with associated hdf5 file into DataFrameArray.

    Also supports optionally iterating or breaking of the file
    into chunks.

    Additional help can be found in the online docs for
    `IO Tools <http://pandas.pydata.org/pandas-docs/stable/io.html>`_.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
    Any valid string path is acceptable. The string could be a URL. Valid
    URL schemes include http, ftp, s3, and file. For file URLs, a host is
    expected. A local file could be: file://localhost/path/to/table.csv.

        If you want to pass in a path object, pandas accepts either
        ``pathlib.Path`` or ``py._path.local.LocalPath``.

        By file-like object, we refer to objects with a ``read()`` method, such as
        a file handler (e.g. via builtin ``open`` function) or ``StringIO``.
        sep : str, default ','
        Delimiter to use. If sep is None, the C engine cannot automatically detect
        the separator, but the Python parsing engine can, meaning the latter will
        be used and automatically detect the separator by Python's builtin sniffer
        tool, ``csv.Sniffer``. In addition, separators longer than 1 character and
        different from ``'\s+'`` will be interpreted as regular expressions and
        will also force the use of the Python parsing engine. Note that regex
        delimiters are prone to ignoring quoted data. Regex example: ``'\r\t'``.
    delimiter : str, default ``None``
        Alias for sep.
    header : int, list of int, default 'infer'
        Row number(s) to use as the column names, and the start of the
        data.  Default behavior is to infer the column names: if no names
        are passed the behavior is identical to ``header=0`` and column
        names are inferred from the first line of the file, if column
        names are passed explicitly then the behavior is identical to
        ``header=None``. Explicitly pass ``header=0`` to be able to
        replace existing names. The header can be a list of integers that
        specify row locations for a multi-index on the columns
        e.g. [0,1,3]. Intervening rows that are not specified will be
        skipped (e.g. 2 in this example is skipped). Note that this
        parameter ignores commented lines and empty lines if
        ``skip_blank_lines=True``, so ``header=0`` denotes the first line of
        data rather than the first line of the file.
    names : array-like, optional
        List of column names to use. If file contains no header row, then you
        should explicitly pass ``header=None``. Duplicates in this list will cause
        a ``UserWarning`` to be issued.
    index_col : int, sequence or bool, optional
        Column to use as the row labels of the DataFrame. If a sequence is given, a
        MultiIndex is used. If you have a malformed file with delimiters at the end
        of each line, you might consider ``index_col=False`` to force pandas to
        not use the first column as the index (row names).
    usecols : list-like or callable, optional
        Return a subset of the columns. If list-like, all elements must either
        be positional (i.e. integer indices into the document columns) or strings
        that correspond to column names provided either by the user in `names` or
        inferred from the document header row(s). For example, a valid list-like
        `usecols` parameter would be ``[0, 1, 2]`` or ``['foo', 'bar', 'baz']``.
        Element order is ignored, so ``usecols=[0, 1]`` is the same as ``[1, 0]``.
        To instantiate a DataFrame from ``data`` with element order preserved use
        ``pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']]`` for columns
        in ``['foo', 'bar']`` order or
        ``pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']]``
        for ``['bar', 'foo']`` order.

        If callable, the callable function will be evaluated against the column
        names, returning names where the callable function evaluates to True. An
        example of a valid callable argument would be ``lambda x: x.upper() in
        ['AAA', 'BBB', 'DDD']``. Using this parameter results in much faster
        parsing time and lower memory usage.
    squeeze : bool, default False
        If the parsed data only contains one column then return a Series.
    prefix : str, optional
        Prefix to add to column numbers when no header, e.g. 'X' for X0, X1, ...
    mangle_dupe_cols : bool, default True
        Duplicate columns will be specified as 'X', 'X.1', ...'X.N', rather than
        'X'...'X'. Passing in False will cause data to be overwritten if there
        are duplicate names in the columns.
    dtype : Type name or dict of column -> type, optional
        Data type for data or columns. E.g. {'a': np.float64, 'b': np.int32,
        'c': 'Int64'}
        Use `str` or `object` together with suitable `na_values` settings
        to preserve and not interpret dtype.
        If converters are specified, they will be applied INSTEAD
        of dtype conversion.
    engine : {'c', 'python'}, optional
        Parser engine to use. The C engine is faster while the python engine is
        currently more feature-complete.
    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can either
        be integers or column labels.
    true_values : list, optional
        Values to consider as True.
    false_values : list, optional
        Values to consider as False.
    skipinitialspace : bool, default False
        Skip spaces after delimiter.
    skiprows : list-like, int or callable, optional
        Line numbers to skip (0-indexed) or number of lines to skip (int)
        at the start of the file.

        If callable, the callable function will be evaluated against the row
        indices, returning True if the row should be skipped and False otherwise.
        An example of a valid callable argument would be ``lambda x: x in [0, 2]``.
    skipfooter : int, default 0
        Number of lines at bottom of file to skip (Unsupported with engine='c').
    nrows : int, optional
        Number of rows of file to read. Useful for reading pieces of large files.
    na_values : scalar, str, list-like, or dict, optional
        Additional strings to recognize as NA/NaN. If dict passed, specific
        per-column NA values.  By default the following values are interpreted as
        NaN: '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
        '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan',
        'null'.
        keep_default_na : bool, default True
        Whether or not to include the default NaN values when parsing the data.
        Depending on whether `na_values` is passed in, the behavior is as follows:

        * If `keep_default_na` is True, and `na_values` are specified, `na_values`
        is appended to the default NaN values used for parsing.
        * If `keep_default_na` is True, and `na_values` are not specified, only
        the default NaN values are used for parsing.
        * If `keep_default_na` is False, and `na_values` are specified, only
        the NaN values specified `na_values` are used for parsing.
        * If `keep_default_na` is False, and `na_values` are not specified, no
        strings will be parsed as NaN.

        Note that if `na_filter` is passed in as False, the `keep_default_na` and
        `na_values` parameters will be ignored.
        na_filter : bool, default True
        Detect missing value markers (empty strings and the value of na_values). In
        data without any NAs, passing na_filter=False can improve the performance
        of reading a large file.
        verbose : bool, default False
        Indicate number of NA values placed in non-numeric columns.
        skip_blank_lines : bool, default True
        If True, skip over blank lines rather than interpreting as NaN values.
        parse_dates : bool or list of int or names or list of lists or dict, default False
        The behavior is as follows:

        * boolean. If True -> try parsing the index.
        * list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
        each as a separate date column.
        * list of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as
        a single date column.
        * dict, e.g. {'foo' : [1, 3]} -> parse columns 1, 3 as date and call
        result 'foo'

        If a column or index contains an unparseable date, the entire column or
        index will be returned unaltered as an object data type. For non-standard
        datetime parsing, use ``pd.to_datetime`` after ``pd.read_csv``

        Note: A fast-path exists for iso8601-formatted dates.
        infer_datetime_format : bool, default False
        If True and `parse_dates` is enabled, pandas will attempt to infer the
        format of the datetime strings in the columns, and if it can be inferred,
        switch to a faster method of parsing them. In some cases this can increase
        the parsing speed by 5-10x.
        keep_date_col : bool, default False
        If True and `parse_dates` specifies combining multiple columns then
        keep the original columns.
        date_parser : function, optional
        Function to use for converting a sequence of string columns to an array of
        datetime instances. The default uses ``dateutil.parser.parser`` to do the
        conversion. Pandas will try to call `date_parser` in three different ways,
        advancing to the next if an exception occurs: 1) Pass one or more arrays
        (as defined by `parse_dates`) as arguments; 2) concatenate (row-wise) the
        string values from the columns defined by `parse_dates` into a single array
        and pass that; and 3) call `date_parser` once for each row using one or
        more strings (corresponding to the columns defined by `parse_dates`) as
        arguments.
        dayfirst : bool, default False
        DD/MM format dates, international and European format.
        iterator : bool, default False
        Return TextFileReader object for iteration or getting chunks with
    ``get_chunk()``.
    chunksize : int, optional
        Return TextFileReader object for iteration.
    See the `IO Tools docs
    <http://pandas.pydata.org/pandas-docs/stable/io.html#io-chunking>`_
        for more information on ``iterator`` and ``chunksize``.
        compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default 'infer'
        For on-the-fly decompression of on-disk data. If 'infer' and
        `filepath_or_buffer` is path-like, then detect compression from the
        following extensions: '.gz', '.bz2', '.zip', or '.xz' (otherwise no
                                                               decompression). If using 'zip', the ZIP file must contain only one data
        file to be read in. Set to None for no decompression.

        .. versionadded:: 0.18.1 support for 'zip' and 'xz' compression.

    thousands : str, optional
    Thousands separator.
    decimal : str, default '.'
    Character to recognize as decimal point (e.g. use ',' for European data).
    lineterminator : str (length 1), optional
    Character to break file into lines. Only valid with C parser.
    quotechar : str (length 1), optional
    The character used to denote the start and end of a quoted item. Quoted
    items can include the delimiter and it will be ignored.
    quoting : int or csv.QUOTE_* instance, default 0
    Control field quoting behavior per ``csv.QUOTE_*`` constants. Use one of
    QUOTE_MINIMAL (0), QUOTE_ALL (1), QUOTE_NONNUMERIC (2) or QUOTE_NONE (3).
    doublequote : bool, default ``True``
    When quotechar is specified and quoting is not ``QUOTE_NONE``, indicate
    whether or not to interpret two consecutive quotechar elements INSIDE a
    field as a single ``quotechar`` element.
    escapechar : str (length 1), optional
    One-character string used to escape other characters.
    comment : str, optional
    Indicates remainder of line should not be parsed. If found at the beginning
    of a line, the line will be ignored altogether. This parameter must be a
    single character. Like empty lines (as long as ``skip_blank_lines=True``),
    fully commented lines are ignored by the parameter `header` but not by
    `skiprows`. For example, if ``comment='#'``, parsing
    ``#empty\na,b,c\n1,2,3`` with ``header=0`` will result in 'a,b,c' being
    treated as the header.
    encoding : str, optional
    Encoding to use for UTF when reading/writing (ex. 'utf-8'). `List of Python
    standard encodings
    <https://docs.python.org/3/library/codecs.html#standard-encodings>`_ .
    dialect : str or csv.Dialect, optional
        If provided, this parameter will override values (default or not) for the
        following parameters: `delimiter`, `doublequote`, `escapechar`,
        `skipinitialspace`, `quotechar`, and `quoting`. If it is necessary to
        override values, a ParserWarning will be issued. See csv.Dialect
        documentation for more details.
        tupleize_cols : bool, default False
        Leave a list of tuples on columns as is (default is to convert to
                                                 a MultiIndex on the columns).

        .. deprecated:: 0.21.0
        This argument will be removed and will always convert to MultiIndex

    error_bad_lines : bool, default True
    Lines with too many fields (e.g. a csv line with too many commas) will by
    default cause an exception to be raised, and no DataFrame will be returned.
        If False, then these "bad lines" will dropped from the DataFrame that is
        returned.
        warn_bad_lines : bool, default True
        If error_bad_lines is False, and warn_bad_lines is True, a warning for each
        "bad line" will be output.
        delim_whitespace : bool, default False
        Specifies whether or not whitespace (e.g. ``' '`` or ``'    '``) will be
        used as the sep. Equivalent to setting ``sep='\s+'``. If this option
        is set to True, nothing should be passed in for the ``delimiter``
        parameter.

        .. versionadded:: 0.18.1 support for the Python parser.

    low_memory : bool, default True
    Internally process the file in chunks, resulting in lower memory use
        while parsing, but possibly mixed type inference.  To ensure no mixed
        types either set False, or specify the type with the `dtype` parameter.
        Note that the entire file is read into a single DataFrame regardless,
        use the `chunksize` or `iterator` parameter to return the data in chunks.
        (Only valid with C parser).
        memory_map : bool, default False
        If a filepath is provided for `filepath_or_buffer`, map the file object
        directly onto memory and access the data directly from there. Using this
        option can improve performance because there is no longer any I/O overhead.
        float_precision : str, optional
        Specifies which converter the C engine should use for floating-point
        values. The options are `None` for the ordinary converter,
        `high` for the high-precision converter, and `round_trip` for the
        round-trip converter.

    Returns
    -------
    DataFrame or TextParser
    A comma-separated values (csv) file is returned as two-dimensional
    data structure with labeled axes.

    See Also
    --------
    to_csv : Write DataFrame to a comma-separated values (csv) file.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_fwf : Read a table of fixed-width formatted lines into DataFrame.

    Examples
    --------
    >>> pd.read_csv('data.csv')  # doctest: +SKIP
    """
    read_csv_args = locals().copy()
    hdf5_path = os.path.splitext(filepath_or_buffer)[0] + ".hdf5"
    df = pd.read_csv(**read_csv_args)
    df_array = []
    with h5py.File(hdf5_path) as f:
        for i, row in df.iterrows():
            data = row.to_dict()
            for col_name, val in row.iteritems():
                try:
                    if val.startswith('hdf5'):
                        data[col_name] = f[val.replace('hdf5_', '')][:]
                except AttributeError:
                    continue
            df_array.append(DataFrameArray(data))
    return pd.concat(df_array).reset_index()
