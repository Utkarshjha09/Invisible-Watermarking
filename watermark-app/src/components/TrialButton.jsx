/**
 * TrialButton Component
 * 
 * A highly customized button component with advanced CSS animations.
 * Features include:
 * - Animated gradient borders that rotate continuously
 * - Shimmer effects on hover
 * - Glow and breathing animations
 * - Glass morphism backdrop
 * - Responsive design
 */

import React from "react";

/**
 * TrialButton Component
 * @param {Object} props - Component props
 * @param {React.ReactNode} props.children - Button text/content
 * @param {...Object} props - Additional button props (onClick, disabled, etc.)
 * @returns {JSX.Element} Animated button with gradient effects
 */
export const TrialButton = ({
  children,
  ...props
}) => {
  return (
    <>
      {/* 
        Embedded CSS Styles
        These styles create the complex gradient animations and effects.
        Using CSS-in-JS approach for component encapsulation.
      */}
      <style>
        {`
        /* Import Google Fonts for consistent typography */
        @import url("https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,500&display=swap");

        /* 
          CSS Custom Properties (CSS Variables) for Animation
          These allow smooth transitions and dynamic color changes
        */
        @property --gradient-angle {
          syntax: "<angle>";
          initial-value: 0deg;
          inherits: false;
        }

        @property --gradient-angle-offset {
          syntax: "<angle>";
          initial-value: 0deg;
          inherits: false;
        }

        @property --gradient-percent {
          syntax: "<percentage>";
          initial-value: 5%;
          inherits: false;
        }

        @property --gradient-shine {
          syntax: "<color>";
          initial-value: white;
          inherits: false;
        }

        /* 
          Main Button Styles
          Creates the base appearance with conic gradient border
        */
        .shiny-custom-styles {
          /* Animation configuration */
          --animation: gradient-angle linear infinite;
          --duration: 3s;
          --shadow-size: 2px;

          /* Conic gradient border background */
          border: 1px solid transparent; /* Required for border-box background-clip */
          background: conic-gradient(
              from calc(var(--gradient-angle) - var(--gradient-angle-offset)),
              transparent,
              #4f46e5 var(--gradient-percent),
              var(--gradient-shine) calc(var(--gradient-percent) * 2),
              #4f46e5 calc(var(--gradient-percent) * 3),
              transparent calc(var(--gradient-percent) * 4)
            )
            border-box;
          box-shadow: inset 0 0 0 1px rgba(26, 26, 26, 0.8); /* Inner shadow */

          /* Transitions for custom properties on hover/focus */
          transition: --gradient-angle-offset 800ms cubic-bezier(0.25, 1, 0.5, 1),
                      --gradient-percent 800ms cubic-bezier(0.25, 1, 0.5, 1),
                      --gradient-shine 800ms cubic-bezier(0.25, 1, 0.5, 1),
                      transform 300ms ease,
                      box-shadow 300ms ease;

          /* Apply base animation */
          animation: var(--animation) var(--duration);
          animation-composition: add; /* Allows multiple animations to combine */
        }

        /* Pseudo-elements common styles */
        .shiny-custom-styles::before,
        .shiny-custom-styles::after,
        .shiny-custom-styles span::before {
          content: "";
          pointer-events: none;
          position: absolute;
          inset-inline-start: 50%;
          inset-block-start: 50%;
          translate: -50% -50%;
          z-index: -1;
        }

        /* Dots pattern for ::before */
        .shiny-custom-styles::before {
          --size: calc(100% - var(--shadow-size) * 3);
          --position: 2px;
          --space: calc(var(--position) * 2);
          width: var(--size);
          height: var(--size);
          background: radial-gradient(
              circle at var(--position) var(--position),
              rgba(255, 255, 255, 0.8) calc(var(--position) / 4),
              transparent 0
            )
            padding-box;
          background-size: var(--space) var(--space);
          background-repeat: space;
          mask-image: conic-gradient(
            from calc(var(--gradient-angle) + 45deg),
            black,
            transparent 10% 90%,
            black
          );
          border-radius: inherit;
          opacity: 0.6;
          z-index: -1;
          animation: var(--animation) var(--duration);
          animation-composition: add;
        }

        /* Inner shimmer for ::after */
        .shiny-custom-styles::after {
          --animation: shimmer linear infinite;
          width: 100%;
          aspect-ratio: 1;
          background: linear-gradient(-50deg, transparent, #4f46e5, transparent);
          mask-image: radial-gradient(circle at bottom, transparent 40%, black);
          opacity: 0.4;
          animation: var(--animation) var(--duration),
                     var(--animation) calc(var(--duration) / 0.4) reverse paused;
          animation-composition: add;
        }

        .shiny-custom-styles span {
          z-index: 1;
          position: relative;
        }

        /* Span's ::before for an additional glow effect */
        .shiny-custom-styles span::before {
          --size: calc(100% + 1rem);
          width: var(--size);
          height: var(--size);
          box-shadow: inset 0 -1ex 2rem 4px #4f46e5;
          opacity: 0;
          transition: opacity 800ms cubic-bezier(0.25, 1, 0.5, 1);
          animation: calc(var(--duration) * 1.5) breathe linear infinite;
        }

        /* Hover/Focus states for animations and custom properties */
        .shiny-custom-styles:is(:hover, :focus-visible) {
          --gradient-percent: 20%;
          --gradient-angle-offset: 95deg;
          --gradient-shine: #8b5cf6; /* Purple shine on hover */
          animation-play-state: running;
          transform: translateY(-2px);
          box-shadow: 0 10px 30px rgba(79, 70, 229, 0.3);
        }

        .shiny-custom-styles:is(:hover, :focus-visible)::before,
        .shiny-custom-styles:is(:hover, :focus-visible)::after {
          animation-play-state: running;
        }

        .shiny-custom-styles:is(:hover, :focus-visible) span::before {
          opacity: 1;
        }

        /* Keyframe animations */
        @keyframes gradient-angle {
          to {
            --gradient-angle: 360deg;
          }
        }

        @keyframes shimmer {
          to {
            rotate: 360deg;
          }
        }

        @keyframes breathe {
          from, to {
            scale: 1;
          }
          50% {
            scale: 1.2;
          }
        }
        `}
      </style>

      <button
        className="backdrop-blur-md font-bold
          shiny-custom-styles
          isolate relative overflow-hidden cursor-pointer
          outline-offset-4
          py-3 px-5
          text-base leading-tight
          rounded-full text-white
          bg-black bg-opacity-60
          active:translate-y-px
          flex items-center justify-center
          min-w-[220px] min-h-[60px]
          transition-all duration-300
        "
        {...props}
      >
        <span className="font-semibold">{children}</span>
      </button>
    </>
  );
};
