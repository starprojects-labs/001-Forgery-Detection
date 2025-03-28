"use client"

import { Button } from "@/components/ui/button"
import { Environment, Float, OrbitControls } from "@react-three/drei"
import { Canvas } from "@react-three/fiber"
import { motion } from "framer-motion"
import { CheckCircle2, Loader2, Trash2, Upload, X } from "lucide-react"
import { Suspense, useCallback, useState } from "react"
import { useDropzone } from "react-dropzone"

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState({
    loader1: false,
    loader2: false,
    loader3: false
  })
  const [uploadComplete, setUploadComplete] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<{
    match_ratio: number;
    match_result: string;
    vgg_prediction: number;
    vgg_result: string;
    final_prediction: string;
    fusion_result: number;
  } | null>(null);

  const [processedImage, setProcessedImage] = useState<{ 
    image: string | null; 
    match_ratio: number | null; 
    forgery_image: string | null; 
}>({
    image: null,
    match_ratio: null,
    forgery_image: null,
});

  const [vggResult, setVggResult] = useState<{
    vgg_prediction: string;
    vgg_result: number;
  } | null>(null);

  const api = "http://127.0.0.1:5000"

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0])
      setUploadComplete(false)
      setResult(null)
      setErrorMessage(null);
      console.log("File selected:", acceptedFiles[0])
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    maxFiles: 1,
    multiple: false,
    accept: {
      "image/*": [],
    },
  })

  const handleAnalyze = async () => {
    if (!file) return

    setUploading((s) => ({ ...s, loader1: true }))
    setResult(null)
    setErrorMessage(null);

    const formData = new FormData()
    formData.append("file", file)
    handleClear()
    setFile(file)

    try {
      const uploadfile = await fetch(api + "/uploadfile", {
        method: "POST",
        body: formData,
      })

      const result = await uploadfile.json()

      if (result.error) {
        setErrorMessage("File format is not supported. Please use another image.");
      } else {
        console.log("File analyzed:", result)
        setResult(result)
      }
    } catch (error) {
      console.error("Analysis failed:", error)
      setErrorMessage("An error occurred. Please try again.");
    } finally {
      setUploading((s) => ({ ...s, loader1: false }))
      setUploadComplete(true)
    }
  }

  const handleKeypoint = async () => {
    if (!file) return;

    setUploading((s) => ({ ...s, loader2: true }));
    setProcessedImage({ image: null, match_ratio: null, forgery_image: null });
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("file", file);
    handleClear();
    setFile(file);

    try {
        const response = await fetch(api + "/keypoint", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            setErrorMessage(data.error || "Failed to process image. Please try again.");
            return;
        }

        if (data.processed_image || data.forgery_image) {
            setProcessedImage({
                image: data.processed_image ? `data:image/jpeg;base64,${data.processed_image}` : null,
                match_ratio: data.match_ratio ?? null,
                forgery_image: data.forgery_image ? `data:image/jpeg;base64,${data.forgery_image}` : null,
            });
        } else {
            alert("No keypoints detected or processing error.");
        }

        console.log("Keypoint analysis:", data);

    } catch (error) {
        console.error("Analysis failed:", error);
        setErrorMessage("An error occurred while processing the image. Please try again.");
    } finally {
        setUploading((s) => ({ ...s, loader2: false }));
        setUploadComplete(true);
    }
};


  const vgganalyze = async () => {
    if (!file) return;

    setUploading((s) => ({ ...s, loader3: true }))
    setVggResult(null);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("file", file);

    handleClear()
    setFile(file)

    try {
      const uploadfile = await fetch(api + "/vggmodel", {
        method: "POST",
        body: formData,
      });

      const result = await uploadfile.json();

      if (result.error) {
        setErrorMessage("File format is not supported. Please use another image.");
      } else {
        console.log("File analyzed:", result);
        setVggResult({
          vgg_prediction: result.vgg_prediction,
          vgg_result: result.vgg_result,
        });
      }
    } catch (error) {
      console.error("Analysis failed:", error);
      setErrorMessage("An error occurred. Please try again.");
    } finally {
      setUploading((s) => ({ ...s, loader3: false }))
      setUploadComplete(true);
    }
  };


  const handleClear = () => {
    setFile(null)
    setUploadComplete(false)
    setResult(null)
    setProcessedImage(null)
    setVggResult(null)
    setErrorMessage(null);
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-violet-800 to-indigo-900 flex flex-col">
      {/* 3D Background */}
      <div className="absolute inset-0 z-0">
        <Canvas>
          <Suspense fallback={null}>
            <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.5} />
            <ambientLight intensity={0.5} />
            <directionalLight position={[10, 10, 5]} intensity={1} />
            <Float speed={4} rotationIntensity={1} floatIntensity={2}>
              <mesh position={[0, 0, -5]} scale={[3, 3, 3]}>
                <torusKnotGeometry args={[1, 0.3, 128, 32]} />
                <meshStandardMaterial color="#4c1d95" wireframe />
              </mesh>
            </Float>
            <Environment preset="city" />
          </Suspense>
        </Canvas>
      </div>

      {/* Content */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center p-8">
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="w-full max-w-md"
        >
          <div {...getRootProps()} className={`
                          p-8 rounded-xl backdrop-blur-md bg-white/10 border-2 
                          ${isDragActive ? "border-purple-400 bg-white/20" : "border-purple-300/50"} 
                          transition-all duration-300 cursor-pointer hover:bg-white/20
                          flex flex-col items-center justify-center text-center
                          h-64
                        `}
          >
            <input {...getInputProps()} />

            {file ? (
              <motion.div
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                className="text-white w-full h-full flex flex-col items-center justify-center"
              >
                <img
                  src={URL.createObjectURL(file) || "/placeholder.svg"}
                  alt="Selected file preview"
                  className="max-h-40 max-w-full rounded-lg shadow-lg object-contain"
                />
                <p className="text-purple-200 break-all mt-2 text-sm">{file.name}</p>
                <p className="text-purple-300 text-xs">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </motion.div>
            ) : (
              <motion.div
                animate={{
                  y: isDragActive ? -10 : 0,
                }}
                className="text-white"
              >
                <Upload className="h-12 w-12 mb-4 mx-auto text-purple-200" />
                <p className="font-medium text-lg">{isDragActive ? "Drop the file here" : "Drag & drop a file here"}</p>
                <p className="text-purple-200 text-sm mt-2">or click to select</p>
              </motion.div>
            )}
          </div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="mt-6 flex justify-center gap-4">
            <Button onClick={handleAnalyze} disabled={!file || uploading.loader1} className="relative overflow-hidden group bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 text-white px-6 py-2 rounded-xl font-medium shadow-lg">
              {uploading.loader1 ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Analyzing...</span>
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  {uploadComplete && !uploading ? <CheckCircle2 className="h-5 w-5" /> : <Upload className="h-5 w-5" />}
                  <span>Fusion</span>
                </span>
              )}
            </Button>

            <Button onClick={vgganalyze} disabled={!file || uploading.loader3} className="relative overflow-hidden group bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 text-white px-6 py-2 rounded-xl font-medium shadow-lg">
              {uploading.loader3 ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Analyzing...</span>
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  {uploadComplete && !uploading ? <CheckCircle2 className="h-5 w-5" /> : <Upload className="h-5 w-5" />}
                  <span>Vgg Model</span>
                </span>
              )}
            </Button>

            <Button onClick={handleKeypoint} disabled={!file || uploading.loader2} className="relative overflow-hidden group bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 text-white px-6 py-2 rounded-xl font-medium shadow-lg">
              {uploading.loader2 ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Analyzing...</span>
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  {uploadComplete && !uploading ? <CheckCircle2 className="h-5 w-5" /> : <Upload className="h-5 w-5" />}
                  <span>Key-Point</span>
                </span>
              )}
            </Button>



            <Button onClick={handleClear} disabled={!file || uploading.loader1 || uploading.loader2 || uploading.loader3} variant="outline" className="bg-white/10 text-white hover:bg-white/20 border-purple-300/50 flex items-center">
              <Trash2 className="h-5 w-5" />
            </Button>
          </motion.div>
          {/* Results Section */}
          {vggResult && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-8 p-6 rounded-xl backdrop-blur-md bg-white/10 border-2 border-purple-300/50 text-center"
            >
              <p className={vggResult.vgg_prediction === "Authentic Image ✅" ? "text-green-400" : "text-red-400"}>
                VGG Prediction: {vggResult.vgg_prediction}
              </p>
              <p className={vggResult.vgg_prediction === "Authentic Image ✅" ? "text-green-400" : "text-red-400"}>
                Confidence Score: {vggResult.vgg_result.toFixed(4)}
              </p>
            </motion.div>
          )}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-8 p-6 rounded-xl backdrop-blur-md bg-white/10 border-2 border-purple-300/50 text-center"
            >
              <div className="text-xl font-small mt-2">
                <p className={`${result.final_prediction === "Authentic Image ✅" ? "text-green-400" : "text-red-400"}`}>
                  Final Ratio: {result.final_prediction}<br />
                  Fusion Prediction : {result.fusion_result}

                </p>
                <p className={`${result.vgg_result === "Authentic Image ✅" ? "text-green-400" : "text-red-400"}`}>
                  Vgg Ratio :{result.vgg_prediction} <br />
                  VGG Prediction: {result.vgg_result}
                </p>
                <p className={`${result.match_result === "Authentic Image ✅" ? "text-green-400" : "text-red-400"}`}>
                  Key Point Ratio : {result.match_ratio} <br />
                  Key Point Prediction: {result.match_result}
                </p>
              </div>
            </motion.div>
          )}
         

         {(processedImage?.image || processedImage?.forgery_image) && (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className="mt-8 p-6 rounded-xl backdrop-blur-md bg-white/10 border-2 border-purple-300/50 text-center"
  >
    {/* Image Container (Flex, Side by Side) */}
    <div className="flex justify-center gap-6">
      {processedImage.image && (
        <div className="flex flex-col items-center">
          <p className="text-sm font-semibold text-white mb-2">Processed Image</p>
          <img 
            src={processedImage.image} 
            alt="Processed" 
            className="rounded-lg shadow-lg w-40 h-40 object-contain"
          />
        </div>
      )}

      {processedImage.forgery_image && (
        <div className="flex flex-col items-center">
          <p className="text-sm font-semibold text-red-400 mb-2">Forgery Image</p>
          <img 
            src={processedImage.forgery_image} 
            alt="Forgery" 
            className="rounded-lg shadow-lg w-40 h-40 object-contain"
          />
        </div>
      )}
    </div>

    {/* Match Ratio Below Images */}
    {processedImage.match_ratio !== null && (
      <p className="mt-4 text-lg font-semibold text-white">
        Match Ratio: {processedImage.match_ratio.toFixed(2)}
      </p>
    )}
  </motion.div>
)}




          {errorMessage && <p className="text-red-400 text-sm mt-4 text-center">{errorMessage}</p>}
        </motion.div>
      </main>
    </div>
  )
}
