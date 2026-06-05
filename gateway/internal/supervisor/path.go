package supervisor

import (
	"fmt"
	"os"
	"path/filepath"
)

// strix/
// ├── bin/
// │   ├── strix          <- os.Executable() — Process A and B share this binary
// │   └── strix_engine   <- Process C
// ├── gateway/
// ├── engine/
// └── model/
//     ├── tokenizer.onnx     <- Tokenizer
//     └── transformer.onnx   <- BERT model

const (
	engineBinName = "strix_engine"
	tokFileName   = "tokenizer.onnx"
	bertFileName  = "transformer.onnx"
)

type ProjectPath struct {
	MainBin   string // strix/bin/strix         - reused to fork Process B
	EngineBin string // strix/bin/strix_engine  - forked as Process C
	TokPath   string // strix/model/tokenizer.onnx
	BertPath  string // strix/model/transfomer.onnx
}

func ResolvePaths() (ProjectPath, error) {
	execPath, pathErr := os.Executable()
	if pathErr != nil {
		return ProjectPath{}, fmt.Errorf(
			"cannot resolve executable path: %w", pathErr,
		)
	}

	execPath, symErr := filepath.EvalSymlinks(execPath)
	if symErr != nil {
		return ProjectPath{}, fmt.Errorf(
			"cannot evaluate symlink on executable path: %w", symErr,
		)
	}

	// Move one level up from bin/ to strix/
	projectRoot := filepath.Clean(filepath.Join(filepath.Dir(execPath), ".."))
	modelFolder := filepath.Join(projectRoot, "model")

	return ProjectPath{
		MainBin:   execPath,
		EngineBin: filepath.Join(projectRoot, "bin", engineBinName),
		TokPath:   filepath.Join(modelFolder, tokFileName),
		BertPath:  filepath.Join(modelFolder, bertFileName),
	}, nil
}
