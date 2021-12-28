#!perl
#
use File::Find qw(find);
my $dir     = '/';
my $pattern = 'dll';
my $t = find sub {print $File::Find::name if /$pattern/}, $dir;
print t;


__DATA__
sub dirwalk {
  local @d_info=();
  local $n = "";
  local $r = "";
  opendir H_DIR,$_[0];
  @dir_info=readdir H_DIR;
  closedir H_DIR;
  foreach $n (@d_info) {
  next if $n=~/^\.*$/;
   if (-d "$_[0]/$n"){
  }
   else {
   next if !($n=~/^.+\.(htm|html)$/);
   ($r,$line) &search_tags("$_[0]/$n");
    if ($r) { push @pages, "$_[0]/$n" if $r;
      }
    }
 }*/