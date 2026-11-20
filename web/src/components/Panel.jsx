import React from "react";
import useReveal from "../useReveal";

export default function Panel({ className = "", children, ...rest }) {
  const [ref, reveal] = useReveal();
  return (
    <section ref={ref} className={`panel ${reveal} ${className}`.trim()} {...rest}>
      {children}
    </section>
  );
}
