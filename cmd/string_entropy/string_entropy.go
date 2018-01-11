package cmd

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"regexp"

	"github.com/willf/entropy"
)

var maxNGramSize = flag.Int("n", 2, "Ngram size")
var train = flag.Bool("train", false, "Set if you want to train")
var predict = flag.Bool("predict", false, "Set if you want to predict")
var infile = flag.String("in", "", "Set for reading")
var outfile = flag.String("out", "", "Set for output")
var modelfile = flag.String("model", "", "Set for model file (dump or read)")
var minLineLength = flag.Uint("min", 0, "Minimum line length, if greater than 0 (for predict)")

func processLine(s string) string {
	compressRegexp := regexp.MustCompile(`\s+`) // should be param?
	text := compressRegexp.ReplaceAllString(s, " ")
	return text
}

func main() {
	flag.Parse()
	ok := (*train || *predict)
	if !ok {
		fmt.Println("Either train or predict must be specified")
		return
	}

	model := entropy.New(*maxNGramSize)

	inf := os.Stdin
	if *infile != "" {
		f2, err := os.Open(*infile)
		if err != nil {
			fmt.Printf("error reading %s\n", *infile)
			return
		}
		inf = f2
		defer inf.Close()
	}

	if *train {

		modf := os.Stdout
		if *modelfile != "" {
			f2, err := os.Create(*modelfile)
			if err != nil {
				fmt.Printf("error creating %s\n", *modelfile)
				return
			}
			modf = f2
			defer modf.Close()
		}
		model.Train(inf)
		model.Dump(modf)
		return
	}

	if *predict {
		modf := os.Stdin
		outf := os.Stdout
		if *modelfile != "" {
			f2, err := os.Open(*modelfile)
			if err != nil {
				fmt.Printf("error reading %s\n", *modelfile)
				return
			}
			modf = f2
		}
		if *outfile != "" {
			f2, err := os.Create(*outfile)
			if err != nil {
				fmt.Printf("error creating %s\n", *outfile)
				return
			}
			outf = f2
			defer outf.Close()
		}

		model = entropy.Read(modf)
		scanner := bufio.NewScanner(inf)
		for scanner.Scan() {
			text := processLine(scanner.Text())
			if uint(len(text)) >= *minLineLength {
				p := model.Predict(text)
				outf.Write([]byte(fmt.Sprintf("%f\t%f\t%v\t%s\n", p.LogProbAverage, p.LogProbTotal, p.NumberOfNGrams, p.Text)))
			}
		}
	}
	// magic_password := "PXKXoyThngGrjCgBLuf2ivrpFFNKA9UgBHrxpLaW"
}
