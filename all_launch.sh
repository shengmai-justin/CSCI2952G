#!/bin/bash
set -e
# THREADS=2
mkdir -p logs

# === Processing A549 ===
./process_srr_to_datafolder.sh A549 GSE82232 SRR3623940 rna-seq 2>&1 | tee logs/SRR3623940_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE82232_SRR3623940_rna-seq
else
    echo "SRR3623940 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh A549 GSE82232 SRR3623932 ribo-seq 2>&1 | tee logs/SRR3623932_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE82232_SRR3623932_ribo-seq
else
    echo "SRR3623932 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing BJ ===
./process_srr_to_datafolder.sh BJ GSE69906 SRR2064024 rna-seq 2>&1 | tee logs/SRR2064024_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE69906_SRR2064024_rna-seq
else
    echo "SRR2064024 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh BJ GSE69906 SRR2064017 ribo-seq 2>&1 | tee logs/SRR2064017_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE69906_SRR2064017_ribo-seq
else
    echo "SRR2064017 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing Brain ===
./process_srr_to_datafolder.sh Brain GSE51424 SRR1562544 rna-seq 2>&1 | tee logs/SRR1562544_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE51424_SRR1562544_rna-seq
else
    echo "SRR1562544 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh Brain GSE51424 SRR1562539 ribo-seq 2>&1 | tee logs/SRR1562539_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE51424_SRR1562539_ribo-seq
else
    echo "SRR1562539 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing CN34 ===
./process_srr_to_datafolder.sh CN34 GSE77292 SRR3129150 rna-seq 2>&1 | tee logs/SRR3129150_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE77292_SRR3129150_rna-seq
else
    echo "SRR3129150 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh CN34 GSE77292 SRR3129148 ribo-seq 2>&1 | tee logs/SRR3129148_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE77292_SRR3129148_ribo-seq
else
    echo "SRR3129148 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing Cybrid ===
./process_srr_to_datafolder.sh Cybrid GSE48933 SRR935456 rna-seq 2>&1 | tee logs/SRR935456_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE48933_SRR935456_rna-seq
else
    echo "SRR935456 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh Cybrid GSE48933 SRR935452 ribo-seq 2>&1 | tee logs/SRR935452_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE48933_SRR935452_ribo-seq
else
    echo "SRR935452 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing H1933 ===
./process_srr_to_datafolder.sh H1933 GSE96716 SRR5350743 rna-seq 2>&1 | tee logs/SRR5350743_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE96716_SRR5350743_rna-seq
else
    echo "SRR5350743 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh H1933 GSE96716 SRR5350745 ribo-seq 2>&1 | tee logs/SRR5350745_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE96716_SRR5350745_ribo-seq
else
    echo "SRR5350745 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing HAP-1 ===
./process_srr_to_datafolder.sh HAP-1 GSE97140 SRR5382428 rna-seq 2>&1 | tee logs/SRR5382428_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE97140_SRR5382428_rna-seq
else
    echo "SRR5382428 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh HAP-1 GSE97140 SRR5382423 ribo-seq 2>&1 | tee logs/SRR5382423_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE97140_SRR5382423_ribo-seq
else
    echo "SRR5382423 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing HEK293T ===
./process_srr_to_datafolder.sh HEK293T GSE52809 SRR1039860 rna-seq 2>&1 | tee logs/SRR1039860_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE52809_SRR1039860_rna-seq
else
    echo "SRR1039860 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh HEK293T GSE52809 SRR1039861 ribo-seq 2>&1 | tee logs/SRR1039861_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE52809_SRR1039861_ribo-seq
else
    echo "SRR1039861 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing HeLa ===
./process_srr_to_datafolder.sh HeLa GSE83493 SRR3680957 rna-seq 2>&1 | tee logs/SRR3680957_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE83493_SRR3680957_rna-seq
else
    echo "SRR3680957 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh HeLa GSE83493 SRR3680966 ribo-seq 2>&1 | tee logs/SRR3680966_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE83493_SRR3680966_ribo-seq
else
    echo "SRR3680966 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing HepG2 ===
./process_srr_to_datafolder.sh HepG2 GSE174419 SRR14525700 rna-seq 2>&1 | tee logs/SRR14525700_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE174419_SRR14525700_rna-seq
else
    echo "SRR14525700 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh HepG2 GSE174419 SRR14525704 ribo-seq 2>&1 | tee logs/SRR14525704_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE174419_SRR14525704_ribo-seq
else
    echo "SRR14525704 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing hESC ===
./process_srr_to_datafolder.sh hESC GSE78959_2 SRR3208858 rna-seq 2>&1 | tee logs/SRR3208858_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE78959_2_SRR3208858_rna-seq
else
    echo "SRR3208858 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh hESC GSE78959_2 SRR3208870 ribo-seq 2>&1 | tee logs/SRR3208870_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE78959_2_SRR3208870_ribo-seq
else
    echo "SRR3208870 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing Huh7 ===
./process_srr_to_datafolder.sh Huh7 GSE69602 SRR2053010 rna-seq 2>&1 | tee logs/SRR2053010_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE69602_SRR2053010_rna-seq
else
    echo "SRR2053010 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh Huh7 GSE69602 SRR2052988 ribo-seq 2>&1 | tee logs/SRR2052988_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE69602_SRR2052988_ribo-seq
else
    echo "SRR2052988 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing K562 ===
./process_srr_to_datafolder.sh K562 GSE153597 SRR12122929 rna-seq 2>&1 | tee logs/SRR12122929_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE153597_SRR12122929_rna-seq
else
    echo "SRR12122929 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh K562 GSE153597 SRR12122935 ribo-seq 2>&1 | tee logs/SRR12122935_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE153597_SRR12122935_ribo-seq
else
    echo "SRR12122935 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing Kidney ===
./process_srr_to_datafolder.sh Kidney GSE59821 SRR2064424 rna-seq 2>&1 | tee logs/SRR2064424_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE59821_SRR2064424_rna-seq
else
    echo "SRR2064424 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh Kidney GSE59821 SRR1528686 ribo-seq 2>&1 | tee logs/SRR1528686_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE59821_SRR1528686_ribo-seq
else
    echo "SRR1528686 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing Macrophages ===
./process_srr_to_datafolder.sh Macrophages GSE66809 SRR1910733 rna-seq 2>&1 | tee logs/SRR1910733_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE66809_SRR1910733_rna-seq
else
    echo "SRR1910733 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh Macrophages GSE66809 SRR1910731 ribo-seq 2>&1 | tee logs/SRR1910731_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE66809_SRR1910731_ribo-seq
else
    echo "SRR1910731 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing MCF7 ===
./process_srr_to_datafolder.sh MCF7 GSE96643 SRR5345617 rna-seq 2>&1 | tee logs/SRR5345617_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE96643_SRR5345617_rna-seq
else
    echo "SRR5345617 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh MCF7 GSE96643 SRR5345621 ribo-seq 2>&1 | tee logs/SRR5345621_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE96643_SRR5345621_ribo-seq
else
    echo "SRR5345621 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing MCF10A ===
./process_srr_to_datafolder.sh MCF10A GSE59817 SRR1528652 rna-seq 2>&1 | tee logs/SRR1528652_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE59817_SRR1528652_rna-seq
else
    echo "SRR1528652 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh MCF10A GSE59817 SRR1528650 ribo-seq 2>&1 | tee logs/SRR1528650_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE59817_SRR1528650_ribo-seq
else
    echo "SRR1528650 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing MDA ===
./process_srr_to_datafolder.sh MDA GSE77315 SRR3129938 rna-seq 2>&1 | tee logs/SRR3129938_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE77315_SRR3129938_rna-seq
else
    echo "SRR3129938 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh MDA GSE77315 SRR3129936 ribo-seq 2>&1 | tee logs/SRR3129936_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE77315_SRR3129936_ribo-seq
else
    echo "SRR3129936 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing MM1.S ===
./process_srr_to_datafolder.sh MM1.S GSE69047 SRR2033082 rna-seq 2>&1 | tee logs/SRR2033082_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE69047_SRR2033082_rna-seq
else
    echo "SRR2033082 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh MM1.S GSE69047 SRR2033088 ribo-seq 2>&1 | tee logs/SRR2033088_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE69047_SRR2033088_ribo-seq
else
    echo "SRR2033088 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing Neuron ===
./process_srr_to_datafolder.sh Neuron GSE90469_2 SRR10103853 rna-seq 2>&1 | tee logs/SRR10103853_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE90469_2_SRR10103853_rna-seq
else
    echo "SRR10103853 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh Neuron GSE90469_2 SRR10103857 ribo-seq 2>&1 | tee logs/SRR10103857_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE90469_2_SRR10103857_ribo-seq
else
    echo "SRR10103857 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing NPC ===
./process_srr_to_datafolder.sh NPC GSE100007_1 SRR5680919 rna-seq 2>&1 | tee logs/SRR5680919_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE100007_1_SRR5680919_rna-seq
else
    echo "SRR5680919 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh NPC GSE100007_1 SRR5680917 ribo-seq 2>&1 | tee logs/SRR5680917_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE100007_1_SRR5680917_ribo-seq
else
    echo "SRR5680917 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing PC3 ===
./process_srr_to_datafolder.sh PC3 GSE35469 SRR403882 rna-seq 2>&1 | tee logs/SRR403882_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE35469_SRR403882_rna-seq
else
    echo "SRR403882 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh PC3 GSE35469 SRR403883 ribo-seq 2>&1 | tee logs/SRR403883_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE35469_SRR403883_ribo-seq
else
    echo "SRR403883 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing PC9 ===
./process_srr_to_datafolder.sh PC9 GSE96716 SRR5350742 rna-seq 2>&1 | tee logs/SRR5350742_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE96716_SRR5350742_rna-seq
else
    echo "SRR5350742 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh PC9 GSE96716 SRR5350744 ribo-seq 2>&1 | tee logs/SRR5350744_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE96716_SRR5350744_ribo-seq
else
    echo "SRR5350744 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing T-ALL ===
./process_srr_to_datafolder.sh T-ALL GSE56887 SRR1248258 rna-seq 2>&1 | tee logs/SRR1248258_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE56887_SRR1248258_rna-seq
else
    echo "SRR1248258 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh T-ALL GSE56887 SRR1248252 ribo-seq 2>&1 | tee logs/SRR1248252_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE56887_SRR1248252_ribo-seq
else
    echo "SRR1248252 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing U2OS ===
./process_srr_to_datafolder.sh U2OS GSE66929 SRR1910466 rna-seq 2>&1 | tee logs/SRR1910466_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE66929_SRR1910466_rna-seq
else
    echo "SRR1910466 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh U2OS GSE66929 SRR1916542 ribo-seq 2>&1 | tee logs/SRR1916542_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE66929_SRR1916542_ribo-seq
else
    echo "SRR1916542 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing WTC_11 ===
./process_srr_to_datafolder.sh WTC_11 GSE131650 SRR9113073 rna-seq 2>&1 | tee logs/SRR9113073_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE131650_SRR9113073_rna-seq
else
    echo "SRR9113073 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh WTC_11 GSE131650 SRR9113067 ribo-seq 2>&1 | tee logs/SRR9113067_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE131650_SRR9113067_ribo-seq
else
    echo "SRR9113067 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing SW480 ===
./process_srr_to_datafolder.sh SW480 GSE196982 SRR18063237 rna-seq 2>&1 | tee logs/SRR18063237_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE196982_SRR18063237_rna-seq
else
    echo "SRR18063237 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh SW480 GSE196982 SRR18063243 ribo-seq 2>&1 | tee logs/SRR18063243_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE196982_SRR18063243_ribo-seq
else
    echo "SRR18063243 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing H9 ===
./process_srr_to_datafolder.sh H9 GSE162050 SRR13120317 rna-seq 2>&1 | tee logs/SRR13120317_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE162050_SRR13120317_rna-seq
else
    echo "SRR13120317 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh H9 GSE162050 SRR13120321 ribo-seq 2>&1 | tee logs/SRR13120321_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE162050_SRR13120321_ribo-seq
else
    echo "SRR13120321 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing MB135iDUX4 ===
./process_srr_to_datafolder.sh MB135iDUX4 GSE178761 SRR14890809 rna-seq 2>&1 | tee logs/SRR14890809_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE178761_SRR14890809_rna-seq
else
    echo "SRR14890809 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh MB135iDUX4 GSE178761 SRR14890797 ribo-seq 2>&1 | tee logs/SRR14890797_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE178761_SRR14890797_ribo-seq
else
    echo "SRR14890797 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing 12T ===
./process_srr_to_datafolder.sh 12T GSE142822 SRR10814103 rna-seq 2>&1 | tee logs/SRR10814103_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE142822_SRR10814103_rna-seq
else
    echo "SRR10814103 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh 12T GSE142822 SRR10814059 ribo-seq 2>&1 | tee logs/SRR10814059_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE142822_SRR10814059_ribo-seq
else
    echo "SRR10814059 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing SH-SY5Y ===
./process_srr_to_datafolder.sh SH-SY5Y GSE155727 SRR12391502 rna-seq 2>&1 | tee logs/SRR12391502_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE155727_SRR12391502_rna-seq
else
    echo "SRR12391502 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh SH-SY5Y GSE155727 SRR12391503 ribo-seq 2>&1 | tee logs/SRR12391503_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE155727_SRR12391503_ribo-seq
else
    echo "SRR12391503 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing Erythroid ===
./process_srr_to_datafolder.sh Erythroid GSE131809 SRR9132374 rna-seq 2>&1 | tee logs/SRR9132374_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE131809_SRR9132374_rna-seq
else
    echo "SRR9132374 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh Erythroid GSE131809 SRR9132358 ribo-seq 2>&1 | tee logs/SRR9132358_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE131809_SRR9132358_ribo-seq
else
    echo "SRR9132358 (ribo-seq) failed – not deleting temp files." >&2
fi

# === Processing Liver ===
./process_srr_to_datafolder.sh Liver GSE112705 SRR6939925 rna-seq 2>&1 | tee logs/SRR6939925_rna.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE112705_SRR6939925_rna-seq
else
    echo "SRR6939925 (rna-seq) failed – not deleting temp files." >&2
fi

./process_srr_to_datafolder.sh Liver GSE112705 SRR6939924 ribo-seq 2>&1 | tee logs/SRR6939924_ribo.log
if [ $? -eq 0 ]; then
    rm -rf /tmp/GSE112705_SRR6939924_ribo-seq
else
    echo "SRR6939924 (ribo-seq) failed – not deleting temp files." >&2
fi
