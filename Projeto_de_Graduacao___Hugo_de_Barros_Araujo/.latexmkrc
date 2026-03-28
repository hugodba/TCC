# Keep generated LaTeX artifacts isolated from source files.
$out_dir = 'build';
$aux_dir = 'build';

# Ensure build directory exists before compilation.
unless (-d $out_dir) {
    mkdir $out_dir or die "Cannot create '$out_dir': $!";
}
