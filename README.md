# Entropy

[![Master Build Status](https://secure.travis-ci.org/willf/entropy.png?branch=master)](https://travis-ci.org/willf/entropy?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/willf/entropy/badge.svg?branch=master)](https://coveralls.io/github/willf/entropy?branch=master)
[![Go Report Card](https://goreportcard.com/badge/github.com/willf/entropy)](https://goreportcard.com/report/github.com/willf/entropy)
[![GoDoc](https://godoc.org/github.com/willf/entropy?status.svg)](http://godoc.org/github.com/willf/entropy)

Really, a character N-gram entropy modeller

This learns a n-gram model on a set of strings, and then can predict
the entropy of other strings.

For example, it has been noted that (good) passwords have high entropy,
and we should be able to use that fact to find (good) passwords in code (where they shoudn't be).

To train:

-  Get some (source) code to train on, and train on it.

The following trains on the 1.7.3 Go distribution code, after removing some crypto files, as well as test files.

The resulting model can be found in the `data` directory.

```bash
find /usr/local/Cellar/go/1.7.3/libexec/src/ | grep "\.go" | grep -v "crypto" | grep -v "_test" | xargs cat > /tmp/go_text
bin/password_entropy -train -in /tmp/go_text -model data/go-3.tsv -ngram_size 3
```

To predict:

- Use the model to predict on some source code, for example,
the source for this program, which has some high-entropy
passwords in it, looking at lines at least 10 characters long (after compressing spaces)

```bash
cat src/cmd/string_entropy/string_entropy.go |  bin/string_entropy -predict -model data/go-3.tsv -min 10  | sort -g | head -5
-16.095489	-997.920341	62	 // magic_password := "PXKXoyThngGrjCgBLuf2ivrpFFNKA9UgBHrxpLaW"
-14.334451	-1576.789572	110	 outf.Write([]byte(fmt.Sprintf("%f\t%f\t%v\t%s\n", p.LogProbAverage, p.LogProbTotal, p.NumberOfNGrams, p.Text)))
-14.186484	-113.491869	8	 modf = f2
-14.186484	-113.491869	8	 modf = f2
-14.107883	-211.618242	15	 model.Dump(modf)
```

Columns are, for each line: average log probability (take negative for entropy), total
log probability, number of ngrams, and the line.

The `Sccanf` line reminds me that format strings always look line line noise, and now we have the science to prove it!
