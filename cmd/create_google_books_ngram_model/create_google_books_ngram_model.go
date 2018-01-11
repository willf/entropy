package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/willf/entropy"
)

var maxNGramSize = flag.Int("n", 2, "Ngram size")
var infile = flag.String("in", "", "Set for reading")
var modelfile = flag.String("model", "", "model file")

func main() {
	flag.Parse()

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
	model.TrainWithMultiplier(inf)
	model.Dump(modf)
	return
}
