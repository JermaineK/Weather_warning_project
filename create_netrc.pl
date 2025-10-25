#!/usr/bin/perl

    use strict;
    use warnings;

    my $urs = "urs.earthdata.nasa.gov";
    my $netrc_file = "$ENV{HOME}/.netrc";
    $| = 1;


    # Ask user for Earthdata Login username

    print "Please enter your Earthdata Login username: ";
    chomp(my $uid = <STDIN>);


    # Ask user for Earthdata Login password

    print "Please enter your Earthdata Login password: ";
    system('/bin/stty', '-echo');
    chomp(my $passwd = <STDIN>);
    system('/bin/stty', 'echo');
    print "\n";

    # Escape '#' and \ (otherwise wget can get confused)

    $passwd =~ s/\\/\\\\/g;
    $passwd =~ s/^#/\\#/;


    # Check to see if a .netrc file already exists. If it does, need to copy
    # across all entries except the URS one (if it exists).

    my @netrc = ("machine $urs login $uid password $passwd");
    if ( -e $netrc_file ) {
        open my $fh1, '<', $netrc_file or die "Could not open existing .netrc file for reading";
        chomp(my @lines = <$fh1>);
        close $fh1;

        foreach (@lines) {
            if ( /^\s*machine\s+$urs\s+login\s+([^\s]+)\s+password\s+([^\s]+)/ ) {
                # This is the line we will be replacing
            }
            else {
                push @netrc, $_;
            }
        }
    }


    # Write the new .netrc file. We use a temporary file first, and only
    # rename it if everything else seems to go ok.

    open my $fh2, '>', "$netrc_file.tmp" or die "Could not create .netrc file";
    foreach (@netrc) {
        print $fh2 "$_\n";
    }
    close $fh2;


    # Move the existing .netrc file (we save a copy just in case)

    if ( -e $netrc_file ) {
        unlink "$netrc_file.old" if -e "$netrc_file.old";
        rename $netrc_file,"$netrc_file.old";
    }

    rename "$netrc_file.tmp", "$netrc_file";
    print "Your .netrc file has been created\n";

    exit(0);