export const panelMotion = {
  initial: { opacity: 0, scale: 0.94, x: -18 },
  animate: { opacity: 1, scale: 1, x: 0 },
  exit: { opacity: 0, scale: 0.97, x: -10 },
  transition: { duration: 0.22, ease: [0.24, 0.72, 0.18, 1] },
};

export const shellMotion = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.28, ease: [0.22, 0.76, 0.22, 1] },
};
