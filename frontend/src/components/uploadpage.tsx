"use client"

import { useState, useCallback, Suspense } from "react"
import { Canvas } from "@react-three/fiber"
import { Environment, Float, OrbitControls } from "@react-three/drei"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Upload, CheckCircle2, Loader2 } from "lucide-react"
import { useDropzone } from "react-dropzone"

export default function UploadPage() {
    const [file, setFile] = useState<File | null>(null)
    const [uploading, setUploading] = useState(false)
    const [uploadComplete, setUploadComplete] = useState(false)

    const api = 'http://localhost:8000'

    const onDrop = useCallback((acceptedFiles: File[]) => {
        if (acceptedFiles.length > 0) {
            setFile(acceptedFiles[0])
            setUploadComplete(false)
        }
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        maxFiles: 1,
        multiple: false,
    })

    const handleUpload = async () => {
        if (!file) return

        setUploading(true)

        // Simulate upload - replace with actual upload logic
        await new Promise((resolve) => setTimeout(resolve, 2000))

        const uploadfile = await fetch(api+'/uploadfile',{
            method: 'POST',
            headers: { 'Content-Type': file.type },
            body: file,
        })

        console.log('File uploaded:', uploadfile)

        setUploading(false)
        setUploadComplete(true)
        // Reset after showing success for a moment
        setTimeout(() => {
            setFile(null)
            setUploadComplete(false)
        }, 3000)
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
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                    className="text-center mb-12"
                >
                    <h1 className="text-4xl md:text-6xl font-bold text-white mb-4 drop-shadow-lg">Welcome to Forgery Detection</h1>
                    <p className="text-xl text-purple-200 max-w-2xl mx-auto">Drop your files below to get started</p>
                </motion.div>

                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.3, duration: 0.5 }}
                    className="w-full max-w-md"
                >
                    <div
                        {...getRootProps()}
                        className={`
              p-8 rounded-xl backdrop-blur-md bg-white/10 border-2 
              ${isDragActive ? "border-purple-400 bg-white/20" : "border-purple-300/50"} 
              transition-all duration-300 cursor-pointer hover:bg-white/20
              flex flex-col items-center justify-center text-center
              h-64
            `}
                    >
                        <input {...getInputProps()} />

                        {file ? (
                            <motion.div initial={{ scale: 0.8 }} animate={{ scale: 1 }} className="text-white">
                                <p className="font-medium text-lg mb-2">File selected:</p>
                                <p className="text-purple-200 break-all">{file.name}</p>
                                <p className="text-purple-300 text-sm mt-2">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
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

                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5 }}
                        className="mt-6 flex justify-center"
                    >
                        <Button
                            onClick={handleUpload}
                            disabled={!file || uploading || uploadComplete}
                            className="relative overflow-hidden group bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 text-white px-8 py-6 rounded-xl text-lg font-medium shadow-lg"
                        >
                            <motion.span
                                animate={{
                                    x: uploading ? 100 : 0,
                                    opacity: uploading ? 0 : 1,
                                }}
                                className="flex items-center gap-2"
                            >
                                {uploadComplete ? (
                                    <>
                                        <CheckCircle2 className="h-5 w-5" />
                                        <span>Uploaded!</span>
                                    </>
                                ) : (
                                    <>
                                        <Upload className="h-5 w-5" />
                                        <span>Upload File</span>
                                    </>
                                )}
                            </motion.span>

                            {uploading && (
                                <motion.span
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="absolute inset-0 flex items-center justify-center"
                                >
                                    <Loader2 className="h-5 w-5 animate-spin" />
                                    <span className="ml-2">Uploading...</span>
                                </motion.span>
                            )}

                            <motion.div
                                className="absolute bottom-0 left-0 h-1 bg-white/30"
                                initial={{ width: 0 }}
                                animate={{
                                    width: uploading ? "100%" : 0,
                                }}
                                transition={{ duration: 2, ease: "linear" }}
                            />
                        </Button>
                    </motion.div>
                </motion.div>
            </main>
        </div>
    )
}