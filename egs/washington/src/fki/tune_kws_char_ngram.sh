#!/usr/bin/env bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/../..";
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export LC_NUMERIC=C;
export PATH="$PWD/../utils:$PATH";
export PYTHONPATH="$PWD/../..:$PYTHONPATH";

help_message="
Usage: ${0##*/} <lats_base_dir> <output_dir>
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 2 ] && echo "$help_message" >&2 && exit 1;

check_all_dirs "$1" || exit 1;
check_all_files -s data/fki/lang/delimiters.txt \
                   data/fki/lang/queries/cv1.txt \
                   data/fki/lang/queries/cv2.txt \
		   data/fki/lang/queries/cv3.txt \
                   data/fki/lang/queries/cv4.txt \
                   data/fki/lang/kws_refs/cv1/va.txt \
                   data/fki/lang/kws_refs/cv2/va.txt \
		   data/fki/lang/kws_refs/cv3/va.txt \
                   data/fki/lang/kws_refs/cv4/va.txt \
		   data/fki/lang/kws_refs/cv1/te.txt \
		   data/fki/lang/kws_refs/cv2/te.txt \
		   data/fki/lang/kws_refs/cv3/te.txt \
		   data/fki/lang/kws_refs/cv4/te.txt || exit 1;

mkdir -p "$2" || exit 1;

for n in 4 5 6 7 8; do
  for cv in cv1 cv2 cv3 cv4; do
    latdir="$1/char_${n}gram/$cv";
    check_all_dirs "$latdir" || exit 1;
    check_all_files -s "data/fki/lm/char_${n}gram/$cv/chars.txt" || exit 1;

    readarray -t delimiters < \
	      <(join -1 1 <(sort -k1b,1 data/fki/lang/delimiters.txt) \
                          <(sort -k1b,1 "data/fki/lm/char_${n}gram/$cv/chars.txt") |
		sort -nk2 | awk '{print $2}') || exit 1;

    logfile="$2/tune_acoustic_prior_scale_kws_${cv}_${n}gram.log";
    [ -s "$logfile" ] ||
    ./src/fki/tune_metric_kws_char.py \
      --queries="data/fki/lang/queries/$cv.txt" \
      --index-type=segment \
      --prior-scale-quant=0.1 \
      --prior-scale-min=0.0 \
      --prior-scale-max=1.5 \
      "data/fki/lm/char_${n}gram/$cv/chars.txt" \
      "data/fki/lang/kws_refs/$cv/va.txt" \
      "$latdir/ps{prior_scale:.1f}/va.lat.ark" \
      "${delimiters[@]}" 2>&1 |
      tee "$logfile" || exit 1;

    readarray -t params < \
      <(tail -n1 "$logfile" |
        gawk '{
          if (match($0, "'\''acoustic_scale'\'': ([0-9.]+)", AM) &&
              match($0, "'\''prior_scale'\'': ([0-9.]+)", PM)) {
            printf("%.2f\n", AM[1]);
            printf("%.1f\n", PM[1]);
          } else {
            print "Wrong last line: "$0 > "/dev/stderr"; exit(1);
          }
        }') || exit 1;
    readarray -t va_map_gap < \
      <(grep "acoustic_scale = ${params[0]}  prior_scale = ${params[1]}" \
	     "$logfile" | gawk '{ print $9; print $12; }') || exit 1;

    check_all_files -s "$latdir/ps${params[1]}/te.lat.ark" || exit 1;
    ./src/fki/compute_kws_metrics_char.py \
      --acoustic-scale="${params[0]}" \
      --queries="data/fki/lang/queries/$cv.txt" \
      --index-type=segment \
      "data/fki/lm/char_${n}gram/$cv/chars.txt" \
      "data/fki/lang/kws_refs/$cv/te.txt" \
      "$latdir/ps${params[1]}/te.lat.ark" \
      "${delimiters[@]}" |
    gawk -v n="$n" -v cv="$cv" -v as="${params[0]}" -v ps="${params[1]}" \
	 -v va_map="${va_map_gap[0]}" -v va_gap="${va_map_gap[1]}" \
    '{
      if (match($0, "'\''mAP'\'': ([0-9.]+)", mAP) &&
          match($0, "'\''gAP'\'': ([0-9.]+)", gAP)) {
        printf("%2-d %s %.9f %.9f %.9f %.9f (as = %.2f  ps = %.1f)\n",
               n, cv, va_map, va_gap, mAP[1], gAP[1], as, ps);
      } else {
        print "Wrong last line: "$0 > "/dev/stderr"; exit(1);
      }
    }' || exit 1;
  done | awk -v n="$n" '{
    print;
    vmAP += $3; vgAP += $4; tmAP += $5; tgAP += $6;
  }END{
    printf("%2-d AVG %.9f %.9f %.9f %.9f\n",
           n, vmAP / NR, vgAP / NR, tmAP / 4, tgAP / 4);
  }'
done;
