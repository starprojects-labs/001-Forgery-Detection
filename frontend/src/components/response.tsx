// // components/YesNoIndicator.tsx
// import { motion } from 'framer-motion';

// interface YesNoIndicatorProps {
//   value: boolean;
// }

// export default function YesNoIndicator({ value }: YesNoIndicatorProps) {
//   return (
//     <motion.div
//       initial={{ opacity: 0, scale: 0.8 }}
//       animate={{ opacity: 1, scale: 1 }}
//       transition={{ duration: 0.5, ease: 'easeOut' }}
//       className={`rounded-2xl shadow-lg p-6 w-48 text-center font-bold text-xl ${
//         value ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
//       }`}
//     >
//       {value ? (
//         <motion.span
//           key="yes"
//           initial={{ y: -10, opacity: 0 }}
//           animate={{ y: 0, opacity: 1 }}
//           transition={{ duration: 0.3 }}
//         >
//           ✅ Yes
//         </motion.span>
//       ) : (
//         <motion.span
//           key="no"
//           initial={{ y: -10, opacity: 0 }}
//           animate={{ y: 0, opacity: 1 }}
//           transition={{ duration: 0.3 }}
//         >
//           ❌ No
//         </motion.span>
//       )}
//     </motion.div>
//   );
// }
